from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, TypeAlias, Union

import numpy as np
import sympy as sp
import torch

from src.tools import np2tc

SpSymbols: TypeAlias = sp.symbols


def normalized_legendre_basis(
    p: int, exact: bool = False, symbol: Union[str, SpSymbols] = "x"
) -> Tuple[Tuple[sp.Expr], Tuple[sp.Expr]]:
    """
    Generate a tuple of normalized Legendre polynomial basis functions
    from order 0 to p (orthonormal in L²[-1,1]) and their first derivatives.
    The normalized orthogonality of these basis functions has been verified.

    Args:
        p (int): Highest polynomial order
        exact (bool): Whether to use exact normalization Expr
        symbol (str or sympy.symbols): Symbol of the independent variable


    Returns:
        return (tuple(basis,basis_deriv)): Tuple of basis and basis_deriv

        basis: Tuple of sympy expressions: (phi_0(x), phi_1(x), ..., phi_p(x))

        basis_deriv: Tuple of sympy expressions: (phi_0'(x), phi_1'(x), ..., phi_p'(x))
    """
    if isinstance(symbol, str):  # Use x by default
        x = sp.symbols(symbol)
    else:  # Use symbol
        x = symbol
    basis = []
    basis_deriv = []
    for n in range(p + 1):
        Pn = sp.legendre(n, x)  # Standard Legendre polynomial P_n(x)
        if exact:  # Exact normalization
            norm_factor = sp.sqrt(sp.Rational(2 * n + 1, 2))
        else:  # float normalization
            norm_factor = sp.sqrt((2 * n + 1) / 2)
        phi_n = norm_factor * Pn
        basis.append(sp.simplify(phi_n))
        basis_deriv.append(sp.simplify(sp.diff(phi_n, x)))
    return tuple(basis), tuple(basis_deriv)


@dataclass(frozen=True)
class Basis(ABC):
    """
    Basis function data container (enforced to use PyTorch).

    Contains: Gauss quadrature points, weights, Vandermonde matrix, derivative matrix,
    L2 projection matrix, boundary evaluation matrix.

    All tensors are torch.float64, computing device can be set.
    """

    x_quad: torch.Tensor  # (N,)
    w: torch.Tensor  # (N,)
    V: torch.Tensor  # (N, p+1)
    V1: torch.Tensor  # (N, p+1)
    P_L2: torch.Tensor  # (p+1, N) V.T @ W
    V1T_W: torch.Tensor  # (p+1, N) V1.T @ W
    B: torch.Tensor  # (2, p+1) B[0]=left endpoint, B[1]=right endpoint
    p: int  # Polynomial order
    N: int  # Number of Gauss quadrature points
    basis_name: str
    device: torch.device

    @classmethod
    @abstractmethod
    def build(cls):
        pass

    def __repr__(self):
        return f"{self.basis_name}(p={self.p}, N={self.N}, device='{self.device}', dtype={self.x_quad.dtype})"


class LegendreBasis1D(Basis):
    """
    One-dimensional normalized Legendre basis function data container
    (enforced to use PyTorch).

    Contains: Gauss quadrature points, weights, Vandermonde matrix, derivative matrix,
    L2 projection matrix, boundary evaluation matrix.

    All tensors are torch.float64, computing device can be set.
    """

    @classmethod
    def build(cls, p: int, N: Optional[int] = None, device: torch.device = torch.device("cpu")):
        """
        Generate one-dimensional Legendre basis function data container.

        Args:
            p (int): Polynomial order
            N (int): Number of Gauss quadrature points
            device (torch.device): Device to place tensors on (default: cpu)

        Returns:
            LegendreBasis1D: One-dimensional Legendre basis function data container
        """
        if N is None:
            N = p + 1

        # 1. Gauss quadrature points and weights (NumPy)
        x_quad_np, w_np = np.polynomial.legendre.leggauss(N)
        x_quad_np = x_quad_np.astype(np.float64)
        w_np = w_np.astype(np.float64)

        # 2. Construct symbolic basis functions
        x = sp.symbols("x")
        basis, basis_deriv = normalized_legendre_basis(p, exact=True, symbol=x)

        # 3. Calculate V, V1 (at Gauss quadrature points)
        V_np = np.empty((N, p + 1), dtype=np.float64)
        V1_np = np.empty((N, p + 1), dtype=np.float64)

        for i in range(p + 1):
            f = sp.lambdify(x, basis[i], modules="numpy")
            f1 = sp.lambdify(x, basis_deriv[i], modules="numpy")
            val = f(x_quad_np)
            val1 = f1(x_quad_np)
            if np.isscalar(val) or val.shape == ():
                val = np.full(N, val, dtype=np.float64)
            if np.isscalar(val1) or val1.shape == ():
                val1 = np.full(N, val1, dtype=np.float64)
            V_np[:, i] = val
            V1_np[:, i] = val1

        P_L2_np = V_np.T @ np.diag(w_np)  # (p+1, N)
        V1T_W_np = V1_np.T @ np.diag(w_np)  # (p+1, N)

        # 4. Calculate boundary matrix B: B[0, i] = phi_i(-1), B[1, i] = phi_i(+1)
        B_np = np.empty((2, p + 1), dtype=np.float64)
        for i in range(p + 1):
            f = sp.lambdify(x, basis[i], modules="numpy")
            B_np[0, i] = f(-1.0)  # Left endpoint x = -1
            B_np[1, i] = f(1.0)  # Right endpoint x = +1

        # 5. Convert to torch.Tensor and move to device
        x_quad = np2tc(x_quad_np, device)
        w = np2tc(w_np, device)
        V = np2tc(V_np, device)
        V1 = np2tc(V1_np, device)
        P_L2 = np2tc(P_L2_np, device)
        V1T_W = np2tc(V1T_W_np, device)
        B = np2tc(B_np, device)

        return cls(
            x_quad=x_quad,
            w=w,
            V=V,
            V1=V1,
            P_L2=P_L2,
            V1T_W=V1T_W,
            B=B,
            p=p,
            N=N,
            device=device,
            basis_name="LegendreBasis1D",
        )
