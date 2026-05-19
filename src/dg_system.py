from abc import ABC, abstractmethod
from enum import Enum
from typing import Union

import matplotlib.pyplot as plt
import torch

from src.field import DGField1D
from src.stepper import DualStepper, ExplicitStepper, RungeKuttaStepper, SimpleExplicitStepper


class EquationOfState(ABC):
    """Base class for Equation of State."""

    @abstractmethod
    def pressure(self, rho: torch.Tensor, u_val: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """Calculate pressure from conservative variables.

        Args:
            rho: Density, shape (...)
            u_val: Velocity, shape (...)
            E: Total energy, shape (...)

        Returns:
            Pressure, shape (...)
        """
        pass

    @abstractmethod
    def sound_speed(self, rho: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Calculate sound speed.

        Args:
            rho: Density, shape (...)
            p: Pressure, shape (...)

        Returns:
            Sound speed, shape (...)
        """
        pass


class IdealGasEOS(EquationOfState):
    """Ideal gas equation of state."""

    def __init__(self, gamma: float = 1.4):
        self.gamma = gamma

    def pressure(self, rho: torch.Tensor, u_val: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        return (self.gamma - 1.0) * (E - 0.5 * rho * u_val**2)

    def sound_speed(self, rho: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # return torch.sqrt(torch.abs(self.gamma * p / rho))
        # p or rho may be negative when oscillation too strong, so try use torch.abs
        # but this may cause non-physical results, so better limiter should be used
        return torch.sqrt(self.gamma * p / rho)


class RiemannSolver(Enum):
    """
    Enum class for Riemann solver.

    Attributes:
        LLF (int): LLF (local_lax_friedrichs) Riemann solver
        ROE (int): Roe Riemann solver
    """

    LLF = 0
    ROE = 1


class SystemOperator1D:
    """
    SystemOperator1D class for 1D system operator.
    """

    def __init__(
        self,
        eos: EquationOfState | None = None,
        riemann_solver: RiemannSolver = RiemannSolver.LLF,
        epsilon_bar: float = 0.1,
    ):
        """
        Init function for DGSystem.

        Args
        ----
        eos (EquationOfState): Equation of state for pressure calculations
        riemann_solver (RiemannSolver): Riemann solver to use
        epsilon_bar (float): Epsilon bar for Harten-Yee entropy correction of Roe solver
        """
        self.eos = eos if eos is not None else IdealGasEOS()
        self.riemann_solver = riemann_solver
        self.epsilon_bar = epsilon_bar

    def get_phy_vars(self, u: torch.Tensor, is2cpu: bool = True) -> torch.Tensor:
        """
        Get primitive variables from conservative variables.

        Args:
            u_quad (torch.Tensor): Input conservative variables of shape (..., 3, :)
                            [..., 0, :] = rho
                            [..., 1, :] = rho * u
                            [..., 2, :] = E
        Returns:
            torch.Tensor: Primitive variables of shape (..., 3, :)
                            [..., 0, :] = rho
                            [..., 1, :] = u
                            [..., 2, :] = p
        """
        rho = u[..., 0, :]
        u_val = u[..., 1, :] / rho
        E = u[..., 2, :]
        p = self.eos.pressure(rho, u_val, E)

        phy_vars = torch.empty_like(u)
        phy_vars[..., 0, :] = rho
        phy_vars[..., 1, :] = u_val
        phy_vars[..., 2, :] = p
        phy_vars = phy_vars.cpu() if is2cpu else phy_vars
        return phy_vars

    def flux_conv_1d(self, u: torch.Tensor, fc: torch.Tensor) -> None:
        """
        Function of 1D convective flux.

        Args:
            u_quad (torch.Tensor): Input conservative variables of shape (..., 3, :)
                            [..., 0, :] = rho
                            [..., 1, :] = rho * u
                            [..., 2, :] = E
            fc (torch.Tensor): Output flux tensor of the same shape as u_quad.
                            This tensor will be overwritten in-place.
        Returns:
            None.
        """

        rho = u[..., 0, :]
        rho_u = u[..., 1, :]
        E = u[..., 2, :]

        # Compute Velocity and Pressure
        u_val = rho_u / rho
        p = self.eos.pressure(rho, u_val, E)

        fc[..., 0, :] = rho_u
        fc[..., 1, :] = rho_u * u_val + p
        fc[..., 2, :] = u_val * (E + p)

    def max_wave_speed_1d(self, u: torch.Tensor) -> torch.Tensor:
        """
        Function of maximum wave speed.

        Args:
            u (torch.Tensor): Input conservative variables of shape (..., 3, :)
                            [..., 0, :] = rho
                            [..., 1, :] = rho * u
                            [..., 2, :] = E
        Returns:
            torch.Tensor: Maximum wave speed of shape (..., :)
        """
        rho = u[..., 0, :]
        rho_u = u[..., 1, :]
        E = u[..., 2, :]

        # Compute Velocity and Pressure
        u_val = rho_u / rho
        p = self.eos.pressure(rho, u_val, E)
        c = self.eos.sound_speed(rho, p)

        # |u| + c
        return torch.abs(u_val) + c

    def local_lax_friedrichs_flux_1d(self, u_lr: torch.Tensor, face_fc: torch.Tensor) -> None:
        """
        Function of local Lax-Friedrichs flux.

        Args:
            u_face_lr (torch.Tensor): Field values at faces' left/right
                shape: (2, num_faces, num_vars)
            face_fc (torch.Tensor): Output flux tensor
                shape: (num_faces, num_vars)
                This tensor will be overwritten in-place.
        """
        u_l = u_lr[0, :, :].unsqueeze(-1)
        u_r = u_lr[1, :, :].unsqueeze(-1)

        # Maximum wave speed on each side
        lambda_l = self.max_wave_speed_1d(u_l)
        lambda_r = self.max_wave_speed_1d(u_r)
        # Local Maximum wave speed
        alpha = torch.maximum(lambda_l, lambda_r, out=lambda_l).unsqueeze(-1)

        # Physical flux at left/right
        fc_l = torch.empty_like(u_l)
        fc_r = torch.empty_like(u_r)
        self.flux_conv_1d(u_l, fc_l)
        self.flux_conv_1d(u_r, fc_r)

        face_fc[:, :] = 0.5 * (fc_l + fc_r - alpha * (u_r - u_l)).squeeze(-1)

    def roe_flux_1d(self, u_lr: torch.Tensor, face_fc: torch.Tensor) -> None:
        """
        Function of Roe flux for 1D Euler equations.

        Args:
            u_lr (torch.Tensor): Field values at faces' left/right
                shape: (2, num_faces, num_vars) where:
                [..., 0, :] = rho
                [..., 1, :] = rho * u
                [..., 2, :] = E
            face_fc (torch.Tensor): Output flux tensor
                shape: (num_faces, num_vars)
                This tensor will be overwritten in-place.
        """
        # Left and right states
        u_l = u_lr[0, :, :].unsqueeze(-1)  # (num_faces, 3, 1)
        u_r = u_lr[1, :, :].unsqueeze(-1)

        # Primitive variables left
        rho_l = u_l[..., 0, :]
        u_l_val = u_l[..., 1, :] / rho_l
        E_l = u_l[..., 2, :]
        p_l = self.eos.pressure(rho_l, u_l_val, E_l)
        H_l = (E_l + p_l) / rho_l

        # Primitive variables right
        rho_r = u_r[..., 0, :]
        u_r_val = u_r[..., 1, :] / rho_r
        E_r = u_r[..., 2, :]
        p_r = self.eos.pressure(rho_r, u_r_val, E_r)
        H_r = (E_r + p_r) / rho_r

        # Roe averages
        sqrt_rho_l = torch.sqrt(rho_l)
        sqrt_rho_r = torch.sqrt(rho_r)
        sqrt_sum = sqrt_rho_l + sqrt_rho_r

        rho_tilde = sqrt_rho_l * sqrt_rho_r
        u_tilde = (sqrt_rho_l * u_l_val + sqrt_rho_r * u_r_val) / sqrt_sum
        H_tilde = (sqrt_rho_l * H_l + sqrt_rho_r * H_r) / sqrt_sum

        # Roe sound speed
        c_tilde_sq = (self.eos.gamma - 1.0) * (H_tilde - 0.5 * u_tilde**2)
        c_tilde = torch.sqrt(c_tilde_sq)

        # Physical flux at left/right
        fc_l = torch.empty_like(u_l)
        fc_r = torch.empty_like(u_r)
        self.flux_conv_1d(u_l, fc_l)
        self.flux_conv_1d(u_r, fc_r)

        # Wave strengths
        drho = rho_r - rho_l
        du = u_r_val - u_l_val
        dp = p_r - p_l

        alpha1 = 0.5 * ((dp - rho_tilde * c_tilde * du) / c_tilde_sq).unsqueeze(-1)
        alpha2 = (drho - dp / c_tilde_sq).unsqueeze(-1)
        alpha3 = 0.5 * ((dp + rho_tilde * c_tilde * du) / c_tilde_sq).unsqueeze(-1)

        # Harten-Yee entropy correction
        # epsilon = epsilon_bar * max(|u| + c) to prevent non-physical expansion shocks
        max_wave_l = torch.abs(u_l_val) + self.eos.sound_speed(rho_l, p_l)
        max_wave_r = torch.abs(u_r_val) + self.eos.sound_speed(rho_r, p_r)
        epsilon = self.epsilon_bar * torch.maximum(max_wave_l, max_wave_r)

        def harten_yee_correction(lambda_val: torch.Tensor) -> torch.Tensor:
            return torch.where(
                torch.abs(lambda_val) >= epsilon,
                torch.abs(lambda_val),
                (lambda_val**2 + epsilon**2) / (2 * epsilon),
            )

        # Absolute eigenvalues with entropy correction
        lambda1 = harten_yee_correction((u_tilde - c_tilde)).unsqueeze(-1)
        lambda2 = harten_yee_correction(u_tilde).unsqueeze(-1)
        lambda3 = harten_yee_correction((u_tilde + c_tilde)).unsqueeze(-1)

        # eigenvectors
        r1 = torch.stack((torch.ones_like(u_tilde), u_tilde - c_tilde, H_tilde - u_tilde * c_tilde), dim=1)
        r2 = torch.stack((torch.ones_like(u_tilde), u_tilde, 0.5 * u_tilde**2), dim=1)
        r3 = torch.stack((torch.ones_like(u_tilde), u_tilde + c_tilde, H_tilde + u_tilde * c_tilde), dim=1)

        # Roe flux
        face_fc[:, :] = (
            0.5 * (fc_l + fc_r) - 0.5 * (alpha1 * lambda1 * r1 + alpha2 * lambda2 * r2 + alpha3 * lambda3 * r3)
        ).squeeze(-1)

    def riemann_solver_1d(self, u_lr: torch.Tensor, face_fc: torch.Tensor) -> None:
        """
        Compute numerical flux based on selected Riemann solver.

        Args:
            u_lr (torch.Tensor): Field values at faces' left/right
                shape: (2, num_faces, num_vars)
            face_fc (torch.Tensor): Output flux tensor
                shape: (num_faces, num_vars)
                This tensor will be overwritten in-place.
        """
        match self.riemann_solver:
            case RiemannSolver.LLF:
                self.local_lax_friedrichs_flux_1d(u_lr, face_fc)
            case RiemannSolver.ROE:
                self.roe_flux_1d(u_lr, face_fc)


class DGSystem1D:
    def __init__(
        self, field: DGField1D, sys_op: SystemOperator1D, stepper: ExplicitStepper | DualStepper | None = None
    ):
        self._mesh = field.mesh
        self._basis = field.basis

        self.field = field
        self.sys_op = sys_op
        self.stepper = stepper
        self.time = 0.0
        self.rhs = torch.zeros_like(field.u)
        self.u_prev = field.u.clone()

    def set_stepper(self, stepper: Union[ExplicitStepper, DualStepper]) -> None:
        self.stepper = stepper

    def cal_rhs_with_u(self, u: torch.Tensor) -> torch.Tensor:
        flux = self.field.get_u_quad_with_u(u)
        self.sys_op.flux_conv_1d(flux, flux)
        # boardcast V1T_W to flux for vectorized matmul
        # unsqueeze(-1) for declaration of column vector in matmul
        # (num_basis, num_quad) @ (num_cells, num_vars, num_quad, 1)
        # -> (num_cells, num_vars, num_basis, 1)
        torch.matmul(self._basis.V1T_W, flux.unsqueeze(-1), out=self.rhs.unsqueeze(-1))
        # face integral
        self.field.set_u_face_lr_with_u(u)
        if self.field.mesh.periodic:
            self.field.set_periodic_bd_cond()
        else:
            self.field.set_transmission_bd_cond()
        self.sys_op.riemann_solver_1d(self.field.u_face_lr, self.field.face_fc)
        self.rhs.index_add_(
            0,
            self._mesh.face_lr[:, 0],
            -torch.mul(
                self._mesh.face_lr_sign[:, 0].unsqueeze(-1).unsqueeze(-1),
                torch.mul(
                    self.field.face_fc[:, :].unsqueeze(-1), self._basis.B[self._mesh.face_lr_map[:, 0], :].unsqueeze(1)
                ),
            ),
        )
        self.rhs.index_add_(
            0,
            self._mesh.face_lr[self._mesh.face_inside, 1],
            -torch.mul(
                self._mesh.face_lr_sign[self._mesh.face_inside, 1].unsqueeze(-1).unsqueeze(-1),
                torch.mul(
                    self.field.face_fc[self._mesh.face_inside, :].unsqueeze(-1),
                    self._basis.B[self._mesh.face_lr_map[self._mesh.face_inside, 1], :].unsqueeze(1),
                ),
            ),
        )
        self.rhs /= self._mesh.jacobians.unsqueeze(-1).unsqueeze(-1)
        if self.rhs.isnan().any():
            raise ValueError("RHS contains nan")
        return self.rhs.clone()

    def plot_field(self):
        x = self.field.field_g_x_quand
        phy_vars = self.sys_op.get_phy_vars(self.field.get_u_quad(), is2cpu=True)
        rho = phy_vars[..., 0, :].flatten().numpy()
        u = phy_vars[..., 1, :].flatten().numpy()
        p = phy_vars[..., 2, :].flatten().numpy()
        plt.clf()
        plt.xlim(0, 1.0)
        plt.ylim(-0.2, 1.2)
        # plt.xlim(0, 1.0)
        # plt.ylim(0.5, 1.5)
        plt.plot(x, rho, label="rho", color="k")
        plt.plot(x, u, label="u", color="r")
        plt.plot(x, p, label="p", color="b")
        plt.legend()
        plt.title(f"time={self.time:3f}s")

    def time_step(self, dt: float) -> None:
        if isinstance(self.stepper, SimpleExplicitStepper):
            self.stepper.step(self.field.u, dt, self.cal_rhs_with_u, out=self.field.u)
        elif isinstance(self.stepper, DualStepper):
            self.stepper.step(self.field.u, dt, out=self.field.u, u_prev=self.u_prev)
        elif isinstance(self.stepper, RungeKuttaStepper):
            self.stepper.step()
        else:
            raise ValueError("stepper must be ExplicitStepper or DualStepper")
        self.time += dt
        print(f"time={self.time:3f}s", end="\r")

    def __repr__(self):
        return f"DGRHS1D(num_cells={self.field.num_cells}, num_vars={self.field.num_vars})"
