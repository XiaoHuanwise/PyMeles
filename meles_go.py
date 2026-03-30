import os
import time
from typing import TypeAlias, Union

os.environ["NUMBA_DISABLE_JIT"] = "1"
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from matplotlib import animation

from src import stepper
from src.basis import LegendreBasis1D
from src.dg_mesh import DGMesh1D
from src.dg_system import DGSystem1D, SystemOperator1D
from src.field import DGField1D

# Set plot font
plt.rcParams["font.family"] = ["SimSun", "Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 12


if __name__ == "__main__":
    # torch.set_num_threads(1)
    process = psutil.Process()
    mem_info = process.memory_info()
    rss = mem_info.rss
    vms = mem_info.vms
    basis = LegendreBasis1D.build(p=2)
    print(basis)
    mesh = DGMesh1D.build_from_equidistant_grid(x_min=0, x_max=1.0, num_cells=100, periodic=True)
    print(mesh)
    sod = lambda x: np.where(x < 0.5, np.array([1.0, 0.0, 1.0 / 0.4]), np.array([0.125, 0.0, 0.1 / 0.4]))
    # max(u+c) = 2.3
    density_perturbation = lambda x: np.array(
        [
            1.0 + 0.2 * np.sin(2 * np.pi * x),
            1.0 * (1.0 + 0.2 * np.sin(2 * np.pi * x)),
            1.0 / 0.4 + 0.5 * (1.0 + 0.2 * np.sin(2 * np.pi * x)) * 1.0 * 1.0,
        ]
    )
    field = DGField1D(basis, mesh, density_perturbation)
    print(field)
    system = DGSystem1D(field, SystemOperator1D())
    print(system)
    # system.plot_field()
    # plt.show()
    implict_stepper = stepper.DITRU2R1Stepper(system.cal_rhs_with_u, field.u)
    explicit_stepper = stepper.SSPRK432Stepper(implict_stepper.F, rtol=1e-6, atol=1e-6)
    dual_stepper = stepper.DualStepper(implict_stepper, explicit_stepper, max_pseudo_steps=100, rtol=1e-3, atol=1e-3)
    system.set_stepper(dual_stepper)
    # system.set_stepper(stepper.SSPRK3Stepper())
    mem_info_add = process.memory_info()
    rss_add = mem_info_add.rss - rss
    vms_add = mem_info_add.vms - vms
    print(f"RSS: {rss_add / 1024**2:.2f} MB")
    print(f"VMS: {vms_add / 1024**2:.2f} MB")
    # system.cal_rhs()
    # plt.ion()
    start_time = time.time()
    for i in range(100):
        system.time_step(0.01)
        # if (i + 1) % 10 == 0:
        #     system.plot_field()
        #     plt.pause(0.01)
    # plt.ioff()
    print(f"time={time.time() - start_time:3f}s")
    system.plot_field()
    plt.show()
