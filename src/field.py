from typing import Callable

import numba as nb
import numpy as np
import torch
from numba import njit

from src.basis import Basis
from src.dg_mesh import DGMesh1D
from src.tools import emptytorch, np2tc, zerostorch


@njit(parallel=True)
def _field_quad_init(
    u_quad: np.ndarray,
    init_func: Callable[[np.float64], np.ndarray],
    cell_vertexs: np.ndarray,
    x_quad: np.ndarray,
):
    """
    Initialize field values at quadrature points.

    Args:
        u_quad (np.ndarray): Field values at quad points (num_cells, num_vars, num_quad)
        init_func (Callable[[np.float64], np.ndarray]): Initialization function
        cell_vertexs (np.ndarray): Cell vertexs (num_cells, 2)
        x_quad (np.ndarray): Quadrature points (num_quad,)
    Return:
        np.ndarray: Field global cood at quadrature points (num_cells * num_quad, num_vars)
    """

    def trans(left, right, X):
        return (right - left) / 2.0 * X + (left + right) / 2.0

    field_g_x_quand = np.empty(u_quad.shape[0] * x_quad.shape[0], dtype=np.float64)

    for i in nb.prange(u_quad.shape[0]):
        g_x_quad = trans(cell_vertexs[i, 0], cell_vertexs[i, 1], x_quad)
        field_g_x_quand[i * x_quad.shape[0] : (i + 1) * x_quad.shape[0]] = g_x_quad
        for j in range(u_quad.shape[2]):
            u_quad[i, :, j] = init_func(g_x_quad[j])

    return field_g_x_quand


class DGField1D:
    """
    DGField1D class for 1D discontinuous Galerkin field.

    Attributes:
        num_cells (int): Number of cells
        num_faces (int): Number of faces
        num_vars (int): Number of variables (default: 3)
        num_basis (int): Number of basis functions
        num_quad (int): Number of quadrature points
        basis (LegendreBasis1D): Basis functions
        mesh (DGMesh1D): Mesh instance
        u (torch.Tensor): Field coefficients of basis functions\
            (num_cells, num_vars, num_basis)
        u_quad (torch.Tensor): Field values at quadrature points\
            (num_cells, num_vars, num_quad)
        u_quad_is_updated (bool): Flag to indicate whether u_quad is updated
        field_g_x_quand (torch.Tensor): Field global cood at quadrature points\
            (num_cells * num_quad)
        u_face_lr (torch.Tensor): Field values at faces' left/right\
            (2, num_faces, num_var)
        face_fc (torch.Tensor): Fluxes of field values at faces' left/right\
            (num_faces, num_var)
        device (torch.device): Device to place tensors on (default: cpu)
    """

    def __init__(
        self,
        basis: Basis,
        mesh: DGMesh1D,
        init_func: Callable[[np.float64], np.ndarray],
        num_vars: int = 3,
        device: torch.device = torch.device("cpu"),
    ):
        if basis.device != device:
            raise ValueError(
                f"Basis and mesh must be on the same device. Basis device: {basis.device}, mesh device: {device}"
            )
        if mesh.device != device:
            raise ValueError(
                f"Basis and mesh must be on the same device. Basis device: {basis.device}, mesh device: {device}"
            )
        self._bd_init = None

        self.num_cells = mesh.num_cells
        self.num_faces = mesh.num_faces
        self.num_vars = num_vars
        self.num_basis = basis.p + 1
        self.num_quad = basis.N
        self.basis = basis
        self.mesh = mesh

        self.u = emptytorch((self.num_cells, self.num_vars, self.num_basis), device)
        self.u_quad = np.empty((self.num_cells, self.num_vars, self.num_quad), dtype=np.float64)
        # .cpu().numpy() may be slow
        # if need preformance, to device operation for mesh and basis should be delay
        self.field_g_x_quand = _field_quad_init(
            self.u_quad,
            nb.njit(nb.float64[:](nb.float64))(init_func),
            self.mesh.cell_vertexs.cpu().numpy(),
            self.basis.x_quad.cpu().numpy(),
        )
        self.u_quad = np2tc(self.u_quad, device)
        # boardcast P_L2 to u_quad for vectorized matmul
        # unsqueeze(-1) for declaration of column vector in matmul
        # (num_basis, num_quad) @ (num_cells, num_vars, num_quad, 1)
        # -> (num_cells, num_vars, num_basis, 1)
        torch.matmul(self.basis.P_L2, self.u_quad.unsqueeze(-1), out=self.u.unsqueeze(-1))

        self.u_face_lr = zerostorch((2, self.num_faces, self.num_vars), device)
        self.face_fc = zerostorch((self.num_faces, self.num_vars), device)
        self.device = device

    def set_u_face_lr(self):
        # boardcast B to u for vectorized matmul
        # unsqueeze(-1) for declaration of column vector in matmul
        # (2, num_basis) @ (num_cells, num_vars, num_basis, 1)
        # -> (num_cells, num_vars, 2, 1)
        u_cell_lr = torch.matmul(self.basis.B, self.u.unsqueeze(-1)).squeeze_(-1)
        self.u_face_lr[0, :, :] = u_cell_lr[self.mesh.face_lr[:, 0], :, self.mesh.face_lr_map[:, 0]]
        self.u_face_lr[1, self.mesh.face_inside, :] = u_cell_lr[
            self.mesh.face_lr[self.mesh.face_inside, 1], :, self.mesh.face_lr_map[self.mesh.face_inside, 1]
        ]

    def set_transmission_bd_cond(self):
        self.u_face_lr[1, self.mesh.face_bd, :] = self.u_face_lr[0, self.mesh.face_bd, :]

    def get_u_quad(self) -> torch.Tensor:
        # boardcast V to u for vectorized matmul
        # unsqueeze(-1) for declaration of column vector in matmul
        # (num_quad, num_basis) @ (num_cells, num_vars, num_basis, 1)
        # -> (num_cells, num_vars, num_quad, 1)
        torch.matmul(self.basis.V, self.u.unsqueeze(-1), out=self.u_quad.unsqueeze(-1))
        return self.u_quad

    def get_u_quad_with_u(self, u: torch.Tensor) -> torch.Tensor:
        # boardcast V to u for vectorized matmul
        # unsqueeze(-1) for declaration of column vector in matmul
        # (num_quad, num_basis) @ (num_cells, num_vars, num_basis, 1)
        # -> (num_cells, num_vars, num_quad, 1)
        torch.matmul(self.basis.V, u.unsqueeze(-1), out=self.u_quad.unsqueeze(-1))
        return self.u_quad

    def add2u(self, addend: torch.Tensor):
        # self.u += addend
        # += is true but add_ is more clear
        self.u.add_(addend)

    def __repr__(self):
        return f"DGField1D(num_cells={self.num_cells}, num_vars={self.num_vars})"
