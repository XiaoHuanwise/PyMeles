from typing import Union

import matplotlib.pyplot as plt
import torch

from src.field import DGField1D
from src.stepper import DualStepper, ExplicitStepper, RungeKuttaStepper, SimpleExplicitStepper


class SystemOperator1D:
    """
    SystemOperator1D class for 1D system operator.

    Attributes:
        flux_conv_1d (Callable[[torch.Tensor, torch.Tensor, float], None]): \
            Function of 1D convective flux
    """

    @staticmethod
    def get_phy_vars(u: torch.Tensor, gamma: float = 1.4, is2cpu: bool = True) -> torch.Tensor:
        """
        Get primitive variables from conservative variables.

        Args:
            u_quad (torch.Tensor): Input conservative variables of shape (..., 3, :)
                            [..., 0, :] = rho
                            [..., 1, :] = rho * u
                            [..., 2, :] = E
            gamma (float): Ratio of specific heats (default: 1.4 for air)
        Returns:
            torch.Tensor: Primitive variables of shape (..., 3, :)
                            [..., 0, :] = rho
                            [..., 1, :] = u
                            [..., 2, :] = p
        """
        phy_vars = torch.empty_like(u)
        phy_vars[..., 0, :] = u[..., 0, :]
        phy_vars[..., 1, :] = u[..., 1, :] / u[..., 0, :]
        phy_vars[..., 2, :] = (gamma - 1.0) * (u[..., 2, :] - 0.5 * u[..., 1, :] * phy_vars[..., 1, :])
        phy_vars = phy_vars.cpu() if is2cpu else phy_vars
        return phy_vars

    @staticmethod
    def flux_conv_1d(u: torch.Tensor, fc: torch.Tensor, gamma: float = 1.4) -> None:
        """
        Function of 1D convective flux.

        Args:
            u_quad (torch.Tensor): Input conservative variables of shape (..., 3, :)
                            [..., 0, :] = rho
                            [..., 1, :] = rho * u
                            [..., 2, :] = E
            fc (torch.Tensor): Output flux tensor of the same shape as u_quad.
                            This tensor will be overwritten in-place.
            gamma (float): Ratio of specific heats (default: 1.4 for air)
        Returns:
            None.
        """

        rho = u[..., 0, :]
        rho_u = u[..., 1, :]
        E = u[..., 2, :]

        # Compute Velocity and Pressure (ideal gas)
        u = rho_u / rho
        p = (gamma - 1.0) * (E - 0.5 * rho_u * u)

        fc[..., 0, :] = rho_u
        fc[..., 1, :] = rho_u * u + p
        fc[..., 2, :] = u * (E + p)

    @staticmethod
    def max_wave_speed_1d(u: torch.Tensor, gamma: float = 1.4) -> torch.Tensor:
        """
        Function of maximum wave speed.

        Args:
            u (torch.Tensor): Input conservative variables of shape (..., 3, :)
                            [..., 0, :] = rho
                            [..., 1, :] = rho * u
                            [..., 2, :] = E
            gamma (float): Ratio of specific heats (default: 1.4 for air)
        Returns:
            torch.Tensor: Maximum wave speed of shape (..., :)
        """
        rho = u[..., 0, :]
        rho_u = u[..., 1, :]
        E = u[..., 2, :]

        # Compute Velocity and Pressure (ideal gas)
        u = rho_u / rho
        p = (gamma - 1.0) * (E - 0.5 * rho_u * u)

        # |u| + sqrt(gamma * p / rho)
        return torch.abs(u) + torch.sqrt(gamma * p / rho)

    @staticmethod
    def local_lax_friedrichs_flux_1d(u_lr: torch.Tensor, face_fc: torch.Tensor, gamma: float = 1.4) -> None:
        """
        Function of local Lax-Friedrichs flux.
        Args:
            u_face_lr (torch.Tensor): Field values at faces' left/right\
            (2, num_faces, num_var)
            face_fc (torch.Tensor): Output flux tensor of same shape as u_lr[0].
                            This tensor will be overwritten in-place.
            gamma (float): Ratio of specific heats (default: 1.4 for air)
        """
        u_l = u_lr[0, :, :].unsqueeze(-1)
        u_r = u_lr[1, :, :].unsqueeze(-1)

        # Maximum wave speed on each side
        lambda_l = SystemOperator1D.max_wave_speed_1d(u_l, gamma)
        lambda_r = SystemOperator1D.max_wave_speed_1d(u_r, gamma)
        # Local Maximum wave speed
        alpha = torch.maximum(lambda_l, lambda_r, out=lambda_l).unsqueeze(-1)

        # Physical flux at left/right
        fc_l = torch.empty_like(u_l)
        fc_r = torch.empty_like(u_r)
        SystemOperator1D.flux_conv_1d(u_l, fc_l, gamma)
        SystemOperator1D.flux_conv_1d(u_r, fc_r, gamma)

        face_fc[:, :] = 0.5 * (fc_l + fc_r - alpha * (u_r - u_l)).squeeze(-1)


class DGSystem1D:
    def __init__(self, field: DGField1D, sys_op: SystemOperator1D, stepper: Union[ExplicitStepper, DualStepper]):
        self._mesh = field.mesh
        self._basis = field.basis

        self.field = field
        self.sys_op = sys_op
        self.stepper = stepper
        self.time = 0.0
        self.rhs = torch.zeros_like(field.u)
        self.u_prev = field.u.clone()

    def cal_rhs_with_u(self, u: torch.Tensor) -> torch.Tensor:
        flux = self.field.get_u_quad_with_u(u)
        self.sys_op.flux_conv_1d(flux, flux)
        # boardcast V1T_W to flux for vectorized matmul
        # unsqueeze(-1) for declaration of column vector in matmul
        # (num_basis, num_quad) @ (num_cells, num_vars, num_quad, 1)
        # -> (num_cells, num_vars, num_basis, 1)
        torch.matmul(self._basis.V1T_W, flux.unsqueeze(-1), out=self.rhs.unsqueeze(-1))
        # face integral
        self.field.set_u_face_lr()
        self.field.set_transmission_bd_cond()
        self.sys_op.local_lax_friedrichs_flux_1d(self.field.u_face_lr, self.field.face_fc)
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
        return self.rhs

    def plot_field(self):
        x = self.field.field_g_x_quand
        phy_vars = self.sys_op.get_phy_vars(self.field.get_u_quad(), is2cpu=True)
        rho = phy_vars[..., 0, :].flatten().numpy()
        u = phy_vars[..., 1, :].flatten().numpy()
        p = phy_vars[..., 2, :].flatten().numpy()
        plt.clf()
        plt.xlim(0, 1.0)
        plt.ylim(-0.2, 1.2)
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
        self.u_prev[...] = self.field.u
        print(f"time={self.time:3f}s", end="\r")

    def __repr__(self):
        return f"DGRHS1D(num_cells={self.field.num_cells}, num_vars={self.field.num_vars})"
