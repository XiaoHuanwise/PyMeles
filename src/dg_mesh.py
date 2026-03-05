import numba as nb
import numpy as np
import torch
from numba import njit

from src.tools import np2tc


@njit(nb.void(nb.float64, nb.float64, nb.int64, nb.float64[:, :], nb.float64[:], nb.int64[:, :]), cache=True)
def _fill_equidistant_grid(  # noqa: PLR0913
    x_min, x_max, num_cells, cell_vertexs, jacobians, cell_face_ids
):
    """
    Fill equidistant grid data using Numba JIT.

    Args:
        x_min:float
        x_max: float
        num_cells: int
        cell_vertexs: (num_cells, 2) array
        jacobians: (num_cells,) array
        cell_face_ids: (num_cells, 2) array
    """

    dx = (x_max - x_min) / num_cells
    for i in range(num_cells):
        xL = x_min + i * dx
        xR = xL + dx
        cell_vertexs[i, 0] = xL
        cell_vertexs[i, 1] = xR
        jacobians[i] = dx / 2.0
        cell_face_ids[i, 0] = i  # Left face index
        cell_face_ids[i, 1] = i + 1  # Right face index


@njit(
    nb.void(nb.int64[:, :], nb.int64, nb.int64, nb.int64[:, :], nb.int64[:, :], nb.int64[:, :], nb.int64[:, :]),
    cache=True,
)
def _build_connectivity(cell_face_ids, num_cells, num_faces, face_lr, face_lr_map, face_lr_sign, cell_neighbors):  # noqa: PLR0913
    """
    Infer face_lr and cell_neighbors based on cell_face_ids.

    Args:
        cell_face_ids: (num_cells, 2) array, dtype=int64
        num_cells: int
        num_faces: int
        face_lr: (num_faces, 2) array, dtype=int64
        face_lr_map: (num_faces, 2) array, dtype=int64
        cell_neighbors: (num_cells, 2) array, dtype=int64
    """

    # Traverse each cell to fill face_lr
    for i in range(num_cells):
        left_face = cell_face_ids[i, 0]
        right_face = cell_face_ids[i, 1]
        face_lr[left_face, 1] = i  # Left face's right cell
        face_lr_map[left_face, 1] = 0
        face_lr_sign[left_face, 1] = -1
        face_lr[right_face, 0] = i  # Right face's left cell
        face_lr_map[right_face, 0] = 1
        face_lr_sign[right_face, 0] = 1

    # Infer cell_neighbors from face_lr
    for f in range(num_faces):
        left_cell = face_lr[f, 0]
        right_cell = face_lr[f, 1]
        if left_cell != -1:
            cell_neighbors[left_cell, 1] = right_cell  # Left cell's right neighbor
        if right_cell != -1:
            cell_neighbors[right_cell, 0] = left_cell  # Right cell's left neighbor


@njit(nb.void(nb.int64[:, :], nb.int64[:, :], nb.int64[:, :], nb.int64[:], nb.int64[:]), cache=True)
def _set_boundary(face_lr, face_lr_map, face_lr_sign, face_inside, face_bd):
    # Set boundary cells to left values of boundary faces
    for i in range(face_bd.shape[0]):
        if face_lr[i, 0] == -1:
            face_lr[i, 0], face_lr[i, 1] = face_lr[i, 1], face_lr[i, 0]
            face_lr_map[i, 0], face_lr_map[i, 1] = face_lr_map[i, 1], face_lr_map[i, 0]
            face_lr_sign[i, 0], face_lr_sign[i, 1] = face_lr_sign[i, 1], face_lr_sign[i, 0]

    face_inside[:] = np.setdiff1d(np.arange(face_lr.shape[0]), face_bd, assume_unique=True)


class DGMesh1D:
    """
    One-dimensional DG mesh data structure.

    All data is initialized with NumPy arrays, supports JIT building topology,
    and can be converted to PyTorch Tensor in-place.
    """

    def __init__(self, num_cells: int):
        """
        Initialize mesh structure (pre-allocate memory).

        Args:
            num_cells (int): Number of cells
            num_faces (int): Number of faces
            cell_vertexs (np.ndarray): (num_cells, 2) array of cell vertexs' coordinates
            jacobians (np.ndarray): (num_cells,) array of cell Jacobians
            cell_face_ids (np.ndarray): (num_cells, 2) array of cells' face IDs
            cell_neighbors (np.ndarray): (num_cells, 2) array of cells' neighbor cell IDs
            face_lr (np.ndarray): (num_faces, 2) array of face left and right cell IDs
            face_lr_map (np.ndarray): (num_faces, 2) array of face left and right cells' local face IDs
            face_lr_sign (np.ndarray): (num_faces, 2) array of face left and right cells' normal sign
            face_bd (np.ndarray): (2,) array of boundary face IDs
            face_inside (np.ndarray): (num_faces - 2,) array of face IDs inside the mesh
            device (torch.device): Device to move tensor's data to
        """
        self.num_cells = num_cells
        self.num_faces = num_cells + 1

        # Cell geometry
        self.cell_vertexs = np.empty((num_cells, 2), dtype=np.float64)  # [xL, xR]
        self.jacobians = np.empty(num_cells, dtype=np.float64)  # J = (xR - xL)/2

        # Topological relationships
        self.cell_face_ids = np.empty(
            (num_cells, 2), dtype=np.int64
        )  # Face IDs of cell boundaries [face_left, face_right]
        self.cell_neighbors = np.full((num_cells, 2), -1, dtype=np.int64)  # Neighbor cell IDs [-1, -1] initialization
        self.face_lr = np.full(
            (self.num_faces, 2), -1, dtype=np.int64
        )  # Left and right cell IDs of faces [-1, -1] initialization
        self.face_lr_map = np.full(
            (self.num_faces, 2), -1, dtype=np.int64
        )  # Left and right cells' relative direction of faces [-1, -1] initialization
        self.face_lr_sign = np.zeros((self.num_faces, 2), dtype=np.int64)  # Left and right cells' normal sign of faces
        self.face_bd = np.empty(2, dtype=np.int64)
        self.face_inside = np.empty(self.num_faces - 2, dtype=np.int64)

        self._filled_cells = 0  # Filled cell counter
        self.device = None

    def set_cell(self, cell_id: int, xL: float, xR: float, face_left: int, face_right: int):
        """
        Fill single cell information.

        Args:
            cell_id (int): Cell ID [0, num_cells)
            xL (float): Left endpoint coordinate
            xR (float): Right endpoint coordinate
            face_left (int): Left face ID
            face_right (int): Right face ID
        """
        if not (0 <= cell_id < self.num_cells):
            raise IndexError(f"cell_id {cell_id} out of range [0, {self.num_cells})")
        if xL >= xR:
            raise ValueError(f"Invalid cell: xL={xL} >= xR={xR}")
        if not (0 <= face_left < self.num_faces):
            raise ValueError(f"face_left {face_left} out of range [0, {self.num_faces})")
        if not (0 <= face_right < self.num_faces):
            raise ValueError(f"face_right {face_right} out of range [0, {self.num_faces})")

        self.cell_vertexs[cell_id, 0] = xL
        self.cell_vertexs[cell_id, 1] = xR
        self.jacobians[cell_id] = (xR - xL) / 2.0
        self.cell_face_ids[cell_id, 0] = face_left
        self.cell_face_ids[cell_id, 1] = face_right
        self._filled_cells += 1

    def build_connectivity(self):
        """
        Infer neighboring cell relationships (using Numba JIT acceleration).

        Must be called after all cells are filled.
        """
        if self._filled_cells != self.num_cells:
            raise RuntimeError("Not all cells have been filled. Call set_cell for all cells first.")

        _build_connectivity(
            self.cell_face_ids,
            self.num_cells,
            self.num_faces,
            self.face_lr,
            self.face_lr_map,
            self.face_lr_sign,
            self.cell_neighbors,
        )

    def set_boundary(self, face_boundary: np.ndarray):
        """
        Set boundary faces.

        Args:
            face_boundary (np.ndarray): Boundary face IDs [left, right]
        """
        self.face_bd[:] = face_boundary
        _set_boundary(self.face_lr, self.face_lr_map, self.face_lr_sign, self.face_inside, self.face_bd)

    @classmethod
    def build_from_equidistant_grid(
        cls,
        x_min: float,
        x_max: float,
        num_cells: int,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Automatically generate DGMesh1D instance from equidistant grid.

        Args:
            x_min (float): Left boundary of solution domain
            x_max (float): Right boundary of solution domain
            num_cells (int): Number of cells (must be > 0)
            device (torch.device): Device to place tensors on (default: cpu)

        Returns:
            DGMesh1D: Mesh instance with filled geometry and built topology
        """
        if num_cells <= 0:
            raise ValueError("num_cells must be positive")
        if x_min >= x_max:
            raise ValueError("x_min must be less than x_max")

        # Create instance
        mesh = cls(num_cells)

        _fill_equidistant_grid(
            x_min,
            x_max,
            num_cells,
            mesh.cell_vertexs,
            mesh.jacobians,
            mesh.cell_face_ids,
        )

        mesh._filled_cells = num_cells

        # Build topological relationships
        mesh.build_connectivity()

        # Set boundary
        mesh.set_boundary(np.array([0, mesh.num_faces - 1], dtype=np.int64))

        mesh.to_torch(device)

        return mesh

    def to_torch(self, device: torch.device = torch.device("cpu")):
        """
        Convert all NumPy arrays to PyTorch Tensors in-place (float64 / int64).

        Args:
            device (torch.device): Target device, default CPU
        """

        self.device = device

        # Geometry data
        self.cell_vertexs = np2tc(self.cell_vertexs, device)
        self.jacobians = np2tc(self.jacobians, device)

        # Topology data
        self.cell_face_ids = np2tc(self.cell_face_ids, device)
        self.cell_neighbors = np2tc(self.cell_neighbors, device)
        self.face_lr = np2tc(self.face_lr, device)
        self.face_lr_map = np2tc(self.face_lr_map, device)
        self.face_lr_sign = np2tc(self.face_lr_sign, device)

        # Boundary data
        self.face_bd = np2tc(self.face_bd, device)
        self.face_inside = np2tc(self.face_inside, device)

    def __repr__(self):
        return f"DGMesh1D(num_cells={self.num_cells})"
