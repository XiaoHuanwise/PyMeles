import functools
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, TypeAlias, Union

import torch

from src import tools

tensor: TypeAlias = torch.Tensor


# ----------------------------------------------------------------------------------------------------------------------
# Explicit Steppers
# ----------------------------------------------------------------------------------------------------------------------


class ExplicitStepper(ABC):
    pass


class SimpleExplicitStepper(ExplicitStepper):
    """
    Base class for explicit steppers.

    Functions
    ---------
    step(u, dt, out=None) -> tensor
        Perform a single step of the ODE.
    """

    @abstractmethod
    def _simple_step_impl(
        self, u: tensor, dt: float, rhs: Callable[[tensor], tensor], *, out: Optional[tensor] = None
    ) -> tensor:
        """
        Perform a single step of the ODE.

        Args
        ----
        u : torch.Tensor
            The current state of the system.
        dt : float
            The time step size.
        rhs : Callable[[torch.Tensor], torch.Tensor]
            The right-hand side of the ODE.
        out : Optional[torch.Tensor]
            The output tensor to store the result, by default None

        Returns
        -------
        tensor
            The state of the system after the step.
        """

        pass

    def step(self, u: tensor, dt: float, rhs: Callable[[tensor], tensor], *, out: Optional[tensor] = None) -> tensor:
        return self._simple_step_impl(u, dt, rhs, out=out)


class FEulerStepper(SimpleExplicitStepper):
    """
    Forward Euler stepper.

    Performs a single step of the ODE using the Forward Euler method.

    Functions
    ---------
    step(u, dt, rhs, out=None) -> tensor
        Perform a single step of the ODE.
    """

    def _simple_step_impl(
        self, u: tensor, dt: float, rhs: Callable[[tensor], tensor], *, out: Optional[tensor] = None
    ) -> tensor:
        if out is None:
            out = torch.empty_like(u)
        if out is not u:
            out.copy_(u)
        out.add_(dt * rhs(u))
        return out


class SSPRK3Stepper(SimpleExplicitStepper):
    """
    SSPRK3 stepper.

    Performs a single step of the ODE using the SSPRK3 method.

    Functions
    ---------
    step(u, dt, rhs, out=None) -> tensor
        Perform a single step of the ODE.
    """

    def _simple_step_impl(
        self, u: tensor, dt: float, rhs: Callable[[tensor], tensor], *, out: Optional[tensor] = None
    ) -> tensor:
        if out is None:
            out = torch.empty_like(u)
        u0 = u
        u1 = u0 + dt * rhs(u0)
        u2 = 3.0 / 4.0 * u0 + 1.0 / 4.0 * u1 + 1.0 / 4.0 * dt * rhs(u1)
        out[...] = 1.0 / 3.0 * u0 + 2.0 / 3.0 * u2 + 2.0 / 3.0 * dt * rhs(u2)
        return out


class RungeKuttaStepper(ExplicitStepper):
    """Base class for explicit Runge-Kutta steppers."""

    # Multiply steps computed from asymptotic behaviour of errors by this.
    SAFETY = 0.9

    MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
    MAX_FACTOR = 5  # Maximum allowed increase in a step size.
    MAX_GROWTH = 10.0

    ALPHA = 0.7
    BETA = 0.4

    C: tensor = NotImplemented
    A: tensor = NotImplemented
    B: tensor = NotImplemented
    E: tensor = NotImplemented
    order: int = NotImplemented
    error_estimator_order: int = NotImplemented
    n_stages: int = NotImplemented

    # Whether the stepper is cell-level or degree_of_freedom-level
    is_cell_level = False

    def __init__(
        self,
        F_ref: tensor,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        min_step: float = 10 * 2**-52,
        max_step: float = float("inf"),
        is_cell_level: bool = False,
    ):
        """
        Init function for Runge-Kutta steppers.

        Args
        ----
        F_ref : tensor
            The reference tensor of F for shape.
        rtol : float
            The relative tolerance, by default 1e-6
        atol : float
            The absolute tolerance, by default 1e-8
        min_step : float
            The minimum step size, by default 10 * 2**-52
        max_step : float
            The maximum step size, by default float("inf")
        """

        self.K = tools.zerostorch((self.n_stages + 1, *F_ref.shape), F_ref.device)
        self.a_view_modal = (-1,) + (1,) * F_ref.dim()

        def __validate_tol(rtol: float, atol: float) -> Tuple[float, float]:
            if (not isinstance(rtol, float)) or (not isinstance(atol, float)) or (rtol < 0) or (atol < 0):
                raise ValueError("rtol and atol must be positive floats")
            # EPS = 2**-52
            # if rtol < 100 * EPS:
            #     raise ValueError(f"rtol must be greater than {100 * EPS}")
            return rtol, atol

        self.rtol, self.atol = __validate_tol(rtol, atol)

        def __validate_step_size(min_step: float, max_step: float) -> Tuple[float, float]:
            if (
                (not isinstance(min_step, float))
                or (not isinstance(max_step, float))
                or (min_step < 0)
                or (max_step < 0)
            ):
                raise ValueError("min_step and max_step must be positive floats")
            return min_step, max_step

        self.min_step, self.max_step = __validate_step_size(min_step, max_step)

        self.error_exponent = 1.0 / (self.error_estimator_order + 1)

        self.C.to(F_ref.device)
        self.A.to(F_ref.device)
        self.B.to(F_ref.device)
        self.E.to(F_ref.device)

        self.is_cell_level = is_cell_level

    def _cal_RMS(self, u: tensor) -> tensor:
        return u.view(-1).norm(p=2) / (u.numel() ** 0.5)

    # def _select_initial_step(self, u0: tensor, f0: tensor, dt: tensor, rhs: Callable[[tensor], tensor]) -> tensor:
    #     # calculate scale factor
    #     scale = self.atol + torch.abs(u0) * self.rtol
    #     # scale = self.atol + self._cal_RMS(u0) * self.rtol
    #     scale[scale < 1e-14] = 1e-14
    #     d0 = u0 / scale
    #     d1 = f0 / scale

    #     # estimate initial step size
    #     h0 = torch.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)  # 1% of the characteristic time scale

    #     # try one step
    #     u1 = u0 + h0 * f0
    #     f1 = rhs(u1)
    #     d2 = (f1 - f0) / scale / h0

    #     # final step
    #     h1 = torch.where(
    #         (d1 <= 1e-15) & (d2 <= 1e-15),
    #         torch.maximum(torch.full_like(h0, 1e-6), h0 * 1e-3),
    #         (0.01 / torch.maximum(d1, d2)) ** self.error_exponent,
    #     )
    #     return torch.minimum(torch.minimum(100 * h0, h1), torch.full_like(h0, self.max_step), out=dt)

    def set_new_state(self, u0: tensor, f0: tensor, dt: tensor, rhs: Callable[[tensor], tensor], phy_dt: float) -> None:
        self.u = u0
        self.f = f0
        self.rhs = rhs
        # self._select_initial_step(u0, f0, dt, rhs)
        single_dt = self._select_initial_step_sdt(u0, f0, rhs)
        self.max_step = phy_dt
        single_dt = min(single_dt, phy_dt)
        dt.fill_(single_dt)
        self.dt = dt
        self.init_dt = dt.clone()
        self.error_norm_prev = torch.ones_like(u0, dtype=u0.dtype)

    def _estimate_error(self, K: tensor, dt: tensor) -> tensor:
        return torch.sum(self.E.view(self.a_view_modal) * K, dim=0) * dt

    def _rk_step(
        self, u: tensor, f: tensor, dt: tensor | float, rhs: Callable[[tensor], tensor]
    ) -> Tuple[tensor, tensor]:
        # Perform a single Runge-Kutta step
        self.K[0, ...] = f
        for s, (a, _) in enumerate(zip(self.A[1:], self.C[1:]), start=1):
            view_a = a.view(self.a_view_modal)
            du = torch.sum(view_a[:s, ...] * self.K[:s, ...], dim=0) * dt
            self.K[s, ...] = rhs(du.add_(u))

        u_new = u + torch.sum(self.B.view(self.a_view_modal) * self.K[:-1, ...], dim=0) * dt
        f_new = rhs(u_new)

        self.K[-1, ...] = f_new

        return u_new, f_new

    def step(self, reject: bool = False) -> None:
        """
        The form of u must be same as the rhs's input.
        the form of out is (u, dt).
        """
        u, f, dt, rhs = self.u, self.f, self.dt, self.rhs

        min_step, max_step = self.min_step, self.max_step
        rtol, atol = self.rtol, self.atol

        step_accepted = False
        step_rejected = False

        # Apply PI control
        while not step_accepted:
            # Apply PI control

            # Perform a single Runge-Kutta step
            u_new, f_new = self._rk_step(u, f, dt, rhs)

            if self.is_cell_level:
                scale = (
                    (
                        atol
                        + torch.maximum(
                            u.flatten(start_dim=-2).norm(p=2, dim=-1) / (u.flatten(start_dim=-2).shape[-1] ** 0.5),
                            u_new.flatten(start_dim=-2).norm(p=2, dim=-1)
                            / (u_new.flatten(start_dim=-2).shape[-1] ** 0.5),
                        )
                        * rtol
                    )
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
                error_norm = (
                    (
                        (self._estimate_error(self.K, dt) / scale).flatten(start_dim=-2).norm(p=2, dim=-1)
                        / (f.flatten(start_dim=-2).shape[-1] ** 0.5)
                    )
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
            else:
                scale = atol + torch.maximum(torch.abs(u), torch.abs(u_new)) * rtol
                error_norm = self._estimate_error(self.K, dt).abs_() / scale

            error_norm[error_norm <= 1e-14] = 1e-14

            factor = (
                self.SAFETY
                * (error_norm ** (-self.ALPHA * self.error_exponent))
                * (self.error_norm_prev ** (self.BETA * self.error_exponent))
            )

            factor[factor > self.MAX_FACTOR] = self.MAX_FACTOR
            factor[factor < self.MIN_FACTOR] = self.MIN_FACTOR
            if error_norm.max() < 1.0:
                if step_rejected:
                    factor[factor > 1.0] = 1.0

                step_accepted = True
            else:
                step_rejected = True

            dt.mul_(factor)
            torch.minimum(dt, self.MAX_GROWTH * self.init_dt, out=dt)
            # torch.maximum(dt, self.init_dt / self.MAX_GROWTH, out=dt)

            dt[dt < min_step] = min_step
            dt[dt > max_step] = max_step

            if not reject:
                break

        self.u[...] = u_new
        self.f = f_new
        self.error_norm_prev = error_norm

    def _select_initial_step_sdt(self, u0: tensor, f0: tensor, rhs: Callable[[tensor], tensor]) -> float:
        # calculate scale factor
        scale = self.atol + self._cal_RMS(u0) * self.rtol
        d0 = self._cal_RMS(u0 / scale)
        d1 = self._cal_RMS(f0 / scale)

        # estimate initial step size
        h0 = 1e-6 if (d0 < 1e-5) or (d1 < 1e-5) else 0.01 * d0 / d1  # 1% of the characteristic time scale

        # try one step
        u1 = u0 + h0 * f0
        f1 = rhs(u1)
        d2 = self._cal_RMS((f1 - f0) / scale) / h0

        # final step
        h1 = torch.where(
            (d1 <= 1e-15) & (d2 < 1e-15),
            torch.maximum(torch.full_like(h0, 1e-6), h0 * 1e-3),
            (0.01 / torch.maximum(d1, d2)) ** self.error_exponent,
        )
        h1 = max(1e-6, h0 * 1e-3) if (d1 <= 1e-15) and (d2 <= 1e-15) else (0.01 / max(d1, d2)) ** self.error_exponent
        self.dt = float(min(100 * h0, h1, self.max_step))
        return self.dt

    def set_new_state_sdt(self, u0: tensor, f0: tensor, rhs: Callable[[tensor], tensor], phy_dt: float) -> None:
        self.u = u0
        self.f = f0
        self.rhs = rhs
        self._select_initial_step_sdt(u0, f0, rhs)
        self.max_step = phy_dt
        self.dt = min(self.dt, self.max_step)
        self.init_dt = self.dt
        self.error_norm_prev = 1.0

    def step_single_dt(self, reject: bool = False) -> None:
        """
        The form of u must be same as the rhs's input.
        the form of out is (u, dt).
        """
        u, f, dt, rhs = self.u, self.f, self.dt, self.rhs

        min_step, max_step = self.min_step, self.max_step
        rtol, atol = self.rtol, self.atol

        # # Initialize accept/reject states
        step_accepted = False
        step_rejected = False

        # Apply PI control
        while not step_accepted:
            # Perform a single Runge-Kutta step
            u_new, f_new = self._rk_step(u, f, dt, rhs)

            scale = atol + torch.maximum(self._cal_RMS(u), self._cal_RMS(u_new)) * rtol
            error_norm = self._cal_RMS(self._estimate_error(self.K, dt) / scale)

            if error_norm < 1.0:
                if error_norm == 0.0:
                    factor = self.MAX_FACTOR
                else:
                    factor = min(
                        self.MAX_FACTOR,
                        self.SAFETY
                        * (error_norm ** (-self.ALPHA * self.error_exponent))
                        * (self.error_norm_prev ** (self.BETA * self.error_exponent)),
                    )

                if step_rejected:
                    factor = min(factor, 1.0)

                dt *= float(factor)

                step_accepted = True
            else:
                dt *= float(
                    max(
                        self.MIN_FACTOR,
                        self.SAFETY
                        * (error_norm ** (-self.ALPHA * self.error_exponent))
                        * (self.error_norm_prev ** (self.BETA * self.error_exponent)),
                    )
                )
                step_rejected = True

            dt = min(dt, self.MAX_GROWTH * self.init_dt)

            dt = max(dt, min_step)
            dt = min(dt, max_step)

            if not reject:
                break

        self.dt = dt
        self.u[...] = u_new
        self.f = f_new
        self.error_norm_prev = error_norm

    def step_set_sdt(self, dt) -> None:
        """
        The form of u must be same as the rhs's input.
        the form of out is (u, dt).
        """
        u, f, rhs = self.u, self.f, self.rhs

        # Perform a single Runge-Kutta step
        u_new, f_new = self._rk_step(u, f, dt, rhs)

        self.u[...] = u_new
        self.f = f_new


class RK32Stepper(RungeKuttaStepper):
    """
    Explicit Runge-Kutta method of order 3(2).

    This uses the Bogacki-Shampine pair of formulas [1]_. The error is controlled
    assuming accuracy of the second-order method, but steps are taken using the
    third-order accurate formula (local extrapolation is done).

    Ref by scipy.integrate._ivp.rk.py

    References
    ----------
    .. [1] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
           Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
    """

    C: tensor = torch.tensor([0, 1 / 2, 3 / 4])
    A: tensor = torch.tensor([[0, 0, 0], [1 / 2, 0, 0], [0, 3 / 4, 0]])
    B: tensor = torch.tensor([2 / 9, 1 / 3, 4 / 9])
    E: tensor = torch.tensor([-5 / 72, 1 / 12, 1 / 9, -1 / 8])

    order: int = 3
    error_estimator_order: int = 2
    n_stages: int = 3


class RK54Stepper(RungeKuttaStepper):
    """
    Explicit Runge-Kutta method of order 5(4).

    This uses the Dormand-Prince pair of formulas [1]_. The error is controlled
    assuming accuracy of the fourth-order method, but steps are taken using the
    fifth-order accurate formula (local extrapolation is done).

    Ref by scipy.integrate._ivp.rk.py

    References
    ----------
    .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
           formulae", Journal of Computational and Applied Mathematics, Vol. 6,
           No. 1, pp. 19-26, 1980.
    """

    order = 5
    error_estimator_order = 4
    n_stages = 6
    C = torch.tensor([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1])
    A = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        ]
    )
    B = torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
    E = torch.tensor([71 / 57600, 0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40])


class SSPRK221Stepper(RungeKuttaStepper):
    """
    Explicit SSP Runge-Kutta method of order (2,2)(1).

    This uses the Shu-Osher pair of formulas [1]_. The error is controlled
    assuming accuracy of the fourth-order method, but steps are taken using the
    fifth-order accurate formula (local extrapolation is done).

    References
    ----------
    .. [1] Imre Fekete, Sidafa Conde, John N. Shadid,
           Embedded pairs for optimal explicit strong stability preserving Runge-Kutta methods,
           Journal of Computational and Applied Mathematics, Volume 412, 2022
    """

    order = 2
    error_estimator_order = 1
    n_stages = 2
    C = torch.tensor([0, 1.0])
    A = torch.tensor(
        [
            [0],
            [1.0],
        ]
    )
    B = torch.tensor([0.5, 0.5])
    # E = torch.tensor([0.5 - 3 / 4, 0.5 - 1 / 4, 0])
    E = torch.tensor([0.5 - 0.694021459207626, 0.5 - 0.305978540792374, 0])


class SSPRK321Stepper(RungeKuttaStepper):
    """
    Explicit SSP Runge-Kutta method of order (3,2)(1).

    This uses the Shu-Osher pair of formulas [1]_. The error is controlled
    assuming accuracy of the fourth-order method, but steps are taken using the
    fifth-order accurate formula (local extrapolation is done).

    References
    ----------
    .. [1] Imre Fekete, Sidafa Conde, John N. Shadid,
           Embedded pairs for optimal explicit strong stability preserving Runge-Kutta methods,
           Journal of Computational and Applied Mathematics, Volume 412, 2022
    """

    order = 2
    error_estimator_order = 1
    n_stages = 3
    C = torch.tensor([0, 0.5, 1.0])
    A = torch.tensor(
        [
            [0, 0],
            [0.5, 0],
            [0.5, 0.5],
        ]
    )
    B = torch.tensor([1 / 3, 1 / 3, 1 / 3])
    # E = torch.tensor([1 / 3 - 4 / 9, 1 / 3 - 1 / 3, 1 / 3 - 2 / 9, 0])
    E = torch.tensor([1 / 3 - 0.635564950337195, 1 / 3 - 0.033488381714827, 1 / 3 - 0.330946667947978, 0])


class SSPRK332Stepper(RungeKuttaStepper):
    """
    Explicit SSP Runge-Kutta method of order (3,3)(2).

    This uses the Shu-Osher pair of formulas [1]_. The error is controlled
    assuming accuracy of the fourth-order method, but steps are taken using the
    fifth-order accurate formula (local extrapolation is done).

    References
    ----------
    .. [1] Imre Fekete, Sidafa Conde, John N. Shadid,
           Embedded pairs for optimal explicit strong stability preserving Runge-Kutta methods,
           Journal of Computational and Applied Mathematics, Volume 412, 2022
    """

    order = 3
    error_estimator_order = 2
    n_stages = 3
    C = torch.tensor([0, 1.0, 0.5])
    A = torch.tensor(
        [
            [0, 0],
            [1.0, 0],
            [0.25, 0.25],
        ]
    )
    B = torch.tensor([1 / 6, 1 / 6, 2 / 3])
    E = torch.tensor([1 / 6 - 0.291485418878409, 1 / 6 - 0.291485418878409, 2 / 3 - 0.417029162243181, 0])


class SSPRK432Stepper(RungeKuttaStepper):
    """
    Explicit SSP Runge-Kutta method of order (4,3)(2).

    This uses the Shu-Osher pair of formulas [1]_. The error is controlled
    assuming accuracy of the fourth-order method, but steps are taken using the
    fifth-order accurate formula (local extrapolation is done).

    References
    ----------
    .. [1] Imre Fekete, Sidafa Conde, John N. Shadid,
           Embedded pairs for optimal explicit strong stability preserving Runge-Kutta methods,
           Journal of Computational and Applied Mathematics, Volume 412, 2022
    """

    order = 3
    error_estimator_order = 2
    n_stages = 4
    C = torch.tensor([0, 0.5, 1, 0.5])
    A = torch.tensor(
        [
            [0, 0, 0],
            [0.5, 0, 0],
            [0.5, 0.5, 0],
            [1 / 6, 1 / 6, 1 / 6],
        ]
    )
    B = torch.tensor([1 / 6, 1 / 6, 1 / 6, 1 / 2])
    # E = torch.tensor([1 / 6 - 1 / 4, 1 / 6 - 1 / 4, 1 / 6 - 1 / 4, 1 / 2 - 1 / 4, 0])
    E = torch.tensor([1 / 6 - 0.138870252716866, 1 / 6 - 0.722259494566267, 1 / 6 - 0.138870252716866, 1 / 2, 0])


# ----------------------------------------------------------------------------------------------------------------------
# Implicit Steppers
# ----------------------------------------------------------------------------------------------------------------------


class ImplicitStepper(ABC):
    """
    Base class for implicit steppers.

    Functions
    ---------
    trhs(u, dt) -> tensor
        The temporal rhs of a single implicit step of the ODE.
    """

    def __init__(self, rhs: Callable[[tensor], tensor]):
        self.rhs: Callable[[tensor], tensor] = rhs

    @abstractmethod
    def _trhs_impl(self, u_new: tensor, u_n: tensor, f_n: tensor | None, u_prev: tensor | None, dt: float) -> tensor:
        pass

    def trhs(self, u_new: tensor, u_n: tensor, f_n: tensor | None, u_prev: tensor | None, dt: float) -> tensor:
        return self._trhs_impl(u_new, u_n, f_n, u_prev, dt)


class BackwardEulerStepper(ImplicitStepper):
    """
    Base class for BackwardEuler stepper.
    """

    F: tensor = NotImplemented  # F_new

    def __init__(self, rhs: Callable[[tensor], tensor], u_ref: tensor):
        super().__init__(rhs)
        self.F = tools.zerostorch(u_ref.shape)

    def _trhs_impl(self, u_new: tensor, u_n: tensor, f_n: tensor | None, u_prev: tensor | None, dt: float) -> tensor:
        del f_n, u_prev
        self.F[...] = (u_n - u_new) / dt + self.rhs(u_new)
        return self.F.clone()


class DITRStepper(ImplicitStepper):
    """
    Base class for DITR stepper.
    """

    c2: float = NotImplemented
    theta: float = NotImplemented  # dt^{n-1}/dt^{n}
    a: tensor = NotImplemented
    b: tensor = NotImplemented
    d: tensor = NotImplemented
    beta: float = NotImplemented
    F: tensor = NotImplemented  # [F_n_c2, F_n_1]

    def __init__(self, rhs: Callable[[tensor], tensor]):
        super().__init__(rhs)

    def _set_b(self, u_ref: tensor, c2: float):
        # [0, b1, b2, b3]
        self.b = torch.tensor(
            [0, 0.5 - 1.0 / (6.0 * c2), 1.0 / (6.0 * c2 * (1.0 - c2)), 0.5 - 1.0 / (6.0 * (1.0 - c2))],
            dtype=u_ref.dtype,
            device=u_ref.device,
        )

    def _mul_P(self, F: tensor) -> None:
        F[0, ...].add_(F[1, ...], alpha=self.beta)


class DITRU2R2Stepper(DITRStepper):
    """
    DITR U2R2 stepper.

    Functions
    ---------
    trhs(u, dt) -> tensor
        The temporal rhs of a single implicit step of the ODE using the DITR U2R2 method.
    """

    def __init__(self, rhs: Callable[[tensor], tensor], u_ref: tensor, c2: float = 0.5, beta: float = 1.0):
        super().__init__(rhs)
        self.c2 = c2
        self._set_b(u_ref, c2)
        self.a = torch.tensor(
            [0, 1.0 - (3.0 * (c2**2) - 2.0 * (c2**3)), 3.0 * (c2**2) - 2.0 * (c2**3)],
            dtype=u_ref.dtype,
            device=u_ref.device,
        )
        self.d = torch.tensor(
            [0, c2 - 2.0 * (c2**2) + (c2**3), -(c2**2) + (c2**3)],
            dtype=u_ref.dtype,
            device=u_ref.device,
        )
        self.beta = beta
        self.F = tools.zerostorch((2, *u_ref.shape), u_ref.device)

    def _trhs_impl(self, u_new: tensor, u_n: tensor, f_n: tensor, u_prev: tensor | None, dt: float) -> tensor:
        """
        The temporal rhs of a single implicit step of the ODE using the DITR U2R2 method.

        The form of u_new, u_n, u_prev  should be tensor[u_{n+c2}, u_{n+1}], u_{n}, u_{n-1}.

        In U2R2, u_prev is not used.
        """

        del u_prev
        u_n_c2, u_n_1 = u_new[0, ...], u_new[1, ...]
        f_n_c2 = self.rhs(u_n_c2)
        f_n_1 = self.rhs(u_n_1)
        # F_n_c2
        self.F[0, ...] = (self.a[1] * u_n + self.a[2] * u_n_1 - u_n_c2) / dt + self.d[1] * f_n + self.d[2] * f_n_1
        # F_n_1
        self.F[1, ...] = (u_n - u_n_1) / dt + self.b[1] * f_n + self.b[2] * f_n_c2 + self.b[3] * f_n_1
        self._mul_P(self.F)
        return self.F.clone()


class DITRU2R1Stepper(DITRStepper):
    """
    DITR U2R1 stepper.

    Functions
    ---------
    trhs(u, dt) -> tensor
        The temporal rhs of a single implicit step of the ODE using the DITR U2R1 method.
    """

    def __init__(self, rhs: Callable[[tensor], tensor], u_ref: tensor, c2: float = 0.5, beta: float = 1.0):
        super().__init__(rhs)
        self.c2 = c2
        self._set_b(u_ref, c2)
        self.a = torch.tensor(
            [0, 1.0 - (2.0 * c2 - (c2**2)), 2.0 * c2 - (c2**2)],
            dtype=u_ref.dtype,
            device=u_ref.device,
        )
        self.d = torch.tensor(
            [0, 0, (c2**2) - c2],
            dtype=u_ref.dtype,
            device=u_ref.device,
        )
        self.beta = beta
        self.F = tools.zerostorch((2, *u_ref.shape), u_ref.device)

    def _trhs_impl(self, u_new: tensor, u_n: tensor, f_n: tensor, u_prev: tensor | None, dt: float) -> tensor:
        """
        The temporal rhs of a single implicit step of the ODE using the DITR U2R1 method.

        The form of u_new, u_n, u_prev  should be tensor[u_{n+c2}, u_{n+1}], u_{n}, u_{n-1}.

        In U2R1, u_prev is not used.
        """

        del u_prev
        u_n_c2, u_n_1 = u_new[0, ...], u_new[1, ...]
        f_n_c2 = self.rhs(u_n_c2)
        f_n_1 = self.rhs(u_n_1)
        # F_n_c2
        self.F[0, ...] = (self.a[1] * u_n + self.a[2] * u_n_1 - u_n_c2) / dt + self.d[2] * f_n_1
        # F_n_1
        self.F[1, ...] = (u_n - u_n_1) / dt + self.b[1] * f_n + self.b[2] * f_n_c2 + self.b[3] * f_n_1
        self._mul_P(self.F)
        return self.F.clone()


class DITRU3R1Stepper(DITRStepper):
    """
    DITR U3R1 stepper.

    Functions
    ---------
    trhs(u, dt) -> tensor
        The temporal rhs of a single implicit step of the ODE using the DITR U2R1 method.
    """

    def __init__(
        self, rhs: Callable[[tensor], tensor], u_ref: tensor, c2: float = 0.5, theta: float = 1.0, beta: float = 1.0
    ):
        super().__init__(rhs)
        self.c2 = c2
        self.theta = theta
        self._set_b(u_ref, c2)
        self.a = torch.tensor(
            [
                -(c2 * ((c2 - 1) ** 2)) / (theta * ((theta + 1) ** 2)),
                ((theta + c2) * ((c2 - 1) ** 2)) / theta,
                (c2 * (-(theta**2) * c2 + 2 * (theta**2) - theta * (c2**2) + 3 * theta - 2 * (c2**2) + 3 * c2))
                / ((theta + 1) ** 2),
            ],
            dtype=u_ref.dtype,
            device=u_ref.device,
        )
        self.d = torch.tensor(
            [0, 0, (c2 * (theta + c2) * (c2 - 1)) / (theta + 1)],
            dtype=u_ref.dtype,
            device=u_ref.device,
        )
        self.beta = beta
        self.F = tools.zerostorch((2, *u_ref.shape), u_ref.device)

    def _trhs_impl(self, u_new: tensor, u_n: tensor, f_n: tensor, u_prev: tensor | None, dt: float) -> tensor:
        """
        The temporal rhs of a single implicit step of the ODE using the DITR U2R1 method.

        The form of u_new, u_n, u_prev  should be tensor[u_{n+c2}, u_{n+1}], u_{n}, u_{n-1}.

        In U2R1, u_prev is not used.
        """

        u_n_c2, u_n_1 = u_new[0, ...], u_new[1, ...]
        f_n_c2 = self.rhs(u_n_c2)
        f_n_1 = self.rhs(u_n_1)
        # F_n_c2
        self.F[0, ...] = (self.a[0] * u_prev + self.a[1] * u_n + self.a[2] * u_n_1 - u_n_c2) / dt + self.d[2] * f_n_1
        # F_n_1
        self.F[1, ...] = (u_n - u_n_1) / dt + self.b[1] * f_n + self.b[2] * f_n_c2 + self.b[3] * f_n_1
        self._mul_P(self.F)
        return self.F.clone()


# ----------------------------------------------------------------------------------------------------------------------
# Dual Steppers
# ----------------------------------------------------------------------------------------------------------------------


class DualStepper:
    def __init__(
        self,
        phy_stepper: ImplicitStepper,
        pseudo_stepper: ExplicitStepper,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        max_pseudo_steps: int = 100,
        is_single_dt: bool = False,
        is_reject: bool = False,
        is_verbose: bool = False,
    ):
        """
        Dual stepper for solving ODEs.
        It is used to solve ODEs with a dual stepper.

        Args
        ----
        phy_stepper : ImplicitStepper
            The physical stepper for solving the ODE.
        pseudo_stepper : ExplicitStepper
            The pseudo stepper for solving the ODE.
        atol : float, optional
            The absolute tolerance for the pseudo stepper, by default 1e-8
        rtol : float, optional
            The relative tolerance for the pseudo stepper, by default 1e-6

        """
        self.phy_stepper = phy_stepper
        self.pseudo_stepper = pseudo_stepper
        self.atol = atol
        self.rtol = rtol
        self.max_pseudo_steps = max_pseudo_steps
        self.REF_STEP = 5  # Magic number from fluent

        if isinstance(phy_stepper, DITRStepper) or isinstance(phy_stepper, BackwardEulerStepper):
            self.u_new = torch.empty_like(phy_stepper.F)
            self.pseudo_dt = torch.empty_like(phy_stepper.F)
        else:
            raise NotImplementedError("Only implemented implicit stepper is supported for dual stepper's phy_stepper.")

        self.u_prev_needed = False

        self.is_single_dt = is_single_dt
        self.is_reject = is_reject
        self.is_verbose = is_verbose

    def _not_converged(self, f_norm: float, f0_norm: float) -> bool:
        return (f_norm / f0_norm > self.rtol) and (f_norm > self.atol)

    def step(
        self,
        u: tensor,
        dt: float,
        *,
        out: Optional[tensor] = None,
        u_prev: Optional[tensor] = None,
    ) -> tensor:
        """
        Perform a single implicit step of the ODE by using the dual stepper.

        Args
        ----
        u : torch.Tensor
            The current state of the system.
        dt : float
            The time step size.
        out : Optional[torch.Tensor]
            The output tensor to store the result, by default None
        u_prev : Optional[torch.Tensor]
            The previous step state of the system.

        Returns
        -------
        tensor
            The state of the system after the step.
        """
        if (u_prev is None) and self.u_prev_needed:
            raise ValueError("u_prev is needed for this stepper.")

        if isinstance(self.phy_stepper, DITRStepper):
            self.u_new[0, ...] = u
            self.u_new[1, ...] = u
        elif isinstance(self.phy_stepper, BackwardEulerStepper):
            self.u_new.copy_(u)
        else:
            raise NotImplementedError("Only implemented implicit stepper is supported for dual stepper's phy_stepper.")

        if isinstance(self.phy_stepper, DITRStepper):
            f_n = self.phy_stepper.rhs(u)
        else:
            f_n = None

        pseudo_rhs = functools.partial(self.phy_stepper.trhs, u_n=u, f_n=f_n, u_prev=u_prev, dt=dt)

        u0 = self.u_new
        f0 = pseudo_rhs(u0)

        if isinstance(self.pseudo_stepper, RungeKuttaStepper):
            # self.pseudo_stepper.set_new_state(u0, f0, self.pseudo_dt, pseudo_rhs, dt)
            self.pseudo_stepper.set_new_state_sdt(u0, f0, pseudo_rhs, dt)
        else:
            raise NotImplementedError("Only RungeKutta stepper is supported for dual stepper's pseudo_stepper.")

        def __cal_norm(x: tensor) -> tensor:
            return torch.norm(x.view(-1), p=float("inf"))

        f0_norm = __cal_norm(f0)
        f_norm = __cal_norm(self.pseudo_stepper.f)
        cnt = 0
        while self._not_converged(f_norm, f0_norm) and cnt < self.max_pseudo_steps:
            if self.is_single_dt:
                self.pseudo_stepper.step_single_dt(self.is_reject)
            else:
                self.pseudo_stepper.step(self.is_reject)
            cnt += 1
            f_norm = __cal_norm(self.pseudo_stepper.f)
            if cnt <= self.REF_STEP:
                f0_norm = max(f0_norm, f_norm)

            if self.is_verbose:
                print(f"{cnt}, {f_norm.item():.3e}, {f_norm.item() / f0_norm.item():.3e}", end="\r")
        if self.is_verbose:
            print("")
            print(f"{cnt}, {f_norm.item():.3e}, {f_norm.item() / f0_norm.item():.3e}")

        if self._not_converged(f_norm, f0_norm) and cnt >= self.max_pseudo_steps:
            print("pseudo stepper touched max steps.")

        u_prev.copy_(u)
        if isinstance(self.phy_stepper, DITRStepper):
            out.copy_(self.u_new[1, ...])
        elif isinstance(self.phy_stepper, BackwardEulerStepper):
            out.copy_(self.u_new)
        else:
            raise NotImplementedError("Only implemented implicit stepper is supported for dual stepper's phy_stepper.")
        return out
