import functools
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, TypeAlias, Union

import torch

from src import tools

tensor: TypeAlias = torch.Tensor


# ----------------------------------------------------------------------------------------------------------------------
# Explicit Steppers
# ----------------------------------------------------------------------------------------------------------------------


class ExplicitStepper(ABC):
    """
    Base class for explicit steppers.

    Functions
    ---------
    step(u, dt, out=None) -> tensor
        Perform a single step of the ODE.
    """

    @abstractmethod
    def _step_impl(
        self,
        u: Union[tensor, Tuple[tensor]],
        dt: Union[float, tensor],
        rhs: Callable[[tensor], tensor],
        *,
        out: Optional[tensor] = None,
    ) -> tensor:
        """
        Perform a single step of the ODE.

        Args
        ----
        u : torch.Tensor or Tunple[torch.Tensor]
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


class FEulerStepper(ExplicitStepper):
    """
    Forward Euler stepper.

    Performs a single step of the ODE using the Forward Euler method.

    Functions
    ---------
    step(u, dt, rhs, out=None) -> tensor
        Perform a single step of the ODE.
    """

    def _step_impl(
        self, u: tensor, dt: float, rhs: Callable[[tensor], tensor], *, out: Optional[tensor] = None
    ) -> tensor:
        if out is None:
            out = torch.empty_like(u)
        if out is not u:
            out.copy_(u)
        out.add_(dt * rhs(u))
        return out


class SSPRK3Stepper(ExplicitStepper):
    """
    SSPRK3 stepper.

    Performs a single step of the ODE using the SSPRK3 method.

    Functions
    ---------
    step(u, dt, rhs, out=None) -> tensor
        Perform a single step of the ODE.
    """

    def _step_impl(
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
    MAX_FACTOR = 10  # Maximum allowed increase in a step size.

    C: tensor = NotImplemented
    A: tensor = NotImplemented
    B: tensor = NotImplemented
    E: tensor = NotImplemented
    order: int = NotImplemented
    error_estimator_order: int = NotImplemented
    n_stages: int = NotImplemented

    def __init__(
        self,
        F_ref: tensor,
        rtol: float = 1e-3,
        atol: float = 1e-6,
        min_step: float = 10 * 2**-52,
        max_step: float = float("inf"),
    ):
        """
        Init function for Runge-Kutta steppers.

        Args
        ----
        F_ref : tensor
            The reference tensor of F for shape.
        rtol : float
            The relative tolerance, by default 1e-3
        atol : float
            The absolute tolerance, by default 1e-6
        min_step : float
            The minimum step size, by default 10 * 2**-52
        max_step : float
            The maximum step size, by default float("inf")
        """

        self.K = tools.zerostorch((self.n_stages + 1, *F_ref.shape), F_ref.device)
        self.a_view_modal = (-1,) + (1,) * F_ref.dim()

        def __validate_tol(self, rtol: float, atol: float) -> Tuple[float, float]:
            if (not isinstance(rtol, float)) or (not isinstance(atol, float)) or (rtol < 0) or (atol < 0):
                raise ValueError("rtol and atol must be positive floats")
            EPS = 2**-52
            if rtol < 100 * EPS:
                raise ValueError(f"rtol must be greater than {100 * EPS}")
            return rtol, atol

        self.rtol, self.atol = __validate_tol(rtol, atol)

        def __validate_step_size(self, min_step: float, max_step: float) -> Tuple[float, float]:
            if (
                (not isinstance(min_step, float))
                or (not isinstance(max_step, float))
                or (min_step < 0)
                or (max_step < 0)
            ):
                raise ValueError("min_step and max_step must be positive floats")
            return min_step, max_step

        self.min_step, self.max_step = __validate_step_size(min_step, max_step)

        self.error_exponent = -1.0 / (self.error_estimator_order + 1)

    def _estimate_error(self, K: tensor, dt: tensor) -> tensor:
        return torch.sum(self.E.view(self.a_view_modal) * K) * dt

    def _step_impl(self, u: tensor, dt: tensor, rhs: Callable[[tensor], tensor]) -> None:
        """
        The form of u must be same as the rhs's input.
        the form of out is (u, dt).
        """
        if isinstance(dt, float):
            dt = torch.tensor(dt, dtype=u[0].dtype, device=u[0].device)
        if dt.shape != self.K[0].shape and dt.numel() != 1:
            raise ValueError("dt must be a tensor with shape (1,) or shape like F_ref or a float")

        min_step, max_step = self.min_step, self.max_step
        rtol, atol = self.rtol, self.atol

        if torch.any(dt < min_step):
            torch.maximum(dt, torch.tensor(min_step, dtype=dt.dtype, device=dt.device), out=dt)
        if torch.any(dt > max_step):
            torch.minimum(dt, torch.tensor(max_step, dtype=dt.dtype, device=dt.device), out=dt)

        # Initialize accept/reject states
        step_accepted = torch.zeros_like(dt, dtype=torch.bool)
        step_rejected = torch.zeros_like(dt, dtype=torch.bool)

        while not step_accepted.all():
            # Check if dt is too small for any unaccepted DOF
            if torch.any(dt < min_step):
                raise ValueError("dt below the min_step without enough accuracy in RK")

            # Perform a single Runge-Kutta step
            self.K[0, ...] = rhs(u)  # TODO: need improvement
            for s, (a, c) in enumerate(zip(self.A[1:], self.C[1:]), start=1):
                view_a = a.view(self.a_view_modal)
                du = torch.sum(view_a[:s, ...] * self.K[:s, ...], dim=0) * dt
                self.K[s, ...] = rhs(du.add_(u))

            u_new = u + torch.sum(self.B.view(self.a_view_modal) * self.K[:-1, ...], dim=0) * dt
            f_new = rhs(u_new)

            self.K[-1, ...] = f_new

            scale = atol + torch.maximum(torch.abs(u), torch.abs(u_new)) * rtol
            error_norm = self._estimate_error(self.K, dt) / scale

            # Only update the DOFs that were not accepted
            mask = not step_accepted
            # Compute accept and reject masks
            mask_accept = mask & (error_norm < 1.0)
            mask_reject = mask & (error_norm >= 1.0)

            # Handle accepted DOFs: adjust dt
            factor_accept = torch.where(
                error_norm == 0.0,
                torch.full_like(error_norm, self.MAX_FACTOR),
                torch.minimum(self.MAX_FACTOR, self.SAFETY * (error_norm**self.error_exponent)),
            )
            # If previously rejected, limit factor <= 1
            factor_accept = torch.where(
                step_rejected, torch.minimum(factor_accept, torch.ones_like(factor_accept)), factor_accept
            )
            dt = torch.where(mask_accept, dt * factor_accept, dt)

            # Handle rejected DOFs: decrease dt
            factor_reject = torch.maximum(self.MIN_FACTOR, self.SAFETY * (error_norm**self.error_exponent))
            dt = torch.where(mask_reject, dt * factor_reject, dt)

            # Update step states
            step_accepted.logical_or_(mask_accept)
            step_rejected.logical_or_(mask_reject)

        u[...] = u_new

        raise NotImplementedError


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
    def trhs(self, u: Tuple[tensor], dt: float) -> tensor:
        pass


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
        F[0, :].add_(F[1, :], alpha=self.beta)

    def get_wrap_trhs(self, dt: float) -> Callable[[tensor], tensor]:
        return functools.partial(self.trhs, dt=dt)


class DITRU2R2Stepper(DITRStepper):
    """
    DITR U2R2 stepper.

    Functions
    ---------
    trhs(u, dt) -> tensor
        The temporal rhs of a single implicit step of the ODE using the DITR U2R2 method.
    """

    def __init__(self, rhs: Callable[[tensor], tensor], u_ref: tensor, c2: float = 0.5, beta: float = 0.5):
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

    def trhs(self, u: Tuple[tensor], dt: float) -> tensor:
        """
        The temporal rhs of a single implicit step of the ODE using the DITR U2R2 method.

        The form of u should be (u_{n}, tensor[u_{n+c2}, u_{n+1}])
        """

        u_n, u_n_c2, u_n_1 = u[0], u[1][0, ...], u[1][1, ...]
        # F_n_c2
        self.F[0, ...] = (
            (self.a[1] * u_n + self.a[2] * u_n_1 - u_n_c2) / dt
            + self.d[1] * self.rhs(u_n)
            + self.d[2] * self.rhs(u_n_1)
        )
        # F_n_1
        self.F[1, ...] = (
            (u_n - u_n_1) / dt + self.b[1] * self.rhs(u_n) + self.b[2] * self.rhs(u_n_c2) + self.b[3] * self.rhs(u_n_1)
        )
        self._mul_P(self.F)
        return self.F
