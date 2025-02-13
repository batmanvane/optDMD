import torch as pt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from flowtorch import DATASETS
from flowtorch.data import FOAMDataloader, mask_box
from flowtorch.analysis import DMD, OptDMD

# increase resolution of plots
mpl.rcParams['figure.dpi'] = 160

# path to the dataset
path = "/Users/robert/WORK/THB/Forschung/optDMD/datasets_minimal/of_cylinder2D_binary"
loader = FOAMDataloader(path)
times = loader.write_times
window_times = [time for time in times if float(time) >= 4.0]

# load vertices, discard z-coordinate, and create a mask
vertices = loader.vertices[:, :2]
mask = mask_box(vertices, lower=[0.1, -1], upper=[0.75, 1])

# assemble data matrix
data_matrix = pt.zeros((mask.sum().item(), len(window_times)), dtype=pt.float32)
for i, time in enumerate(window_times):
    # load the vorticity vector field, take the z-component [:, 2], and apply the mask
    data_matrix[:, i] = pt.masked_select(loader.load_snapshot("vorticity", time)[:, 2], mask)

dt = float(times[1]) - float(times[0])
dmd = DMD(data_matrix, dt=dt, rank=19)
#dmd = OptDMD(data_matrix, dt=dt, rank=19)

print(dmd)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
t = pt.linspace(0, 2 * np.pi, 100)
ax.plot(pt.cos(t), pt.sin(t), ls="--", color="k", lw=2)
ax.scatter(dmd.eigvals.real, dmd.eigvals.imag, zorder=7)
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_xlabel(r"$Re(\lambda)$")
ax.set_ylabel(r"$Im(\lambda)$")
for i, val in enumerate(dmd.eigvals):
    index = "{" + f"{i}" + "}"
    ax.annotate(r"$\lambda_{:s}$".format(index), (val.real*1.13, val.imag*1.13), ha='center', va="center")
plt.show()

fig, ax = plt.subplots()
amplitude = dmd.amplitude.real.abs()
_, ind = pt.topk(amplitude, 11)
ax.bar(dmd.frequency, amplitude)
for i, (a, f) in enumerate(zip(amplitude, dmd.frequency)):
    if i in ind[1:]:
        text = r"$f_{:s} = {:2.2f} Hz$".format("{"+f"{i}"+"}", f)
        ax.annotate(text, (f, a+80), ha="center", fontsize=8, rotation=90)
ax.axvline(0.0, ls="--", c="k")
ax.set_xlabel("frequency")
ax.set_ylabel("amplitude")
plt.show()

fig, axarr = plt.subplots(4, 1, sharex=True)

times_num = [float(t) for t in window_times]
modes = [1, 3, 5, 11]
for i, m in enumerate(modes):
    axarr[i].plot(times_num, dmd.dynamics[m].real, lw=1)
    axarr[i].set_ylabel(f"mode {m}")
axarr[-1].set_xlabel(r"$t$ in $s$")
axarr[-1].set_xlim(4.0, 10.0)
axarr[0].set_title("time dynamics")
plt.show()

x = pt.masked_select(vertices[:, 0], mask)
y = pt.masked_select(vertices[:, 1], mask)

def add_mode(ax, mode, title, every=4):
    ax.tricontourf(x[::every], y[::every], mode[::every], levels=15, cmap="jet")
    ax.tricontour(x[::every], y[::every], mode[::every], levels=15, linewidths=0.1, colors='k')
    ax.add_patch(plt.Circle((0.2, 0.2), 0.05, color='k'))
    ax.set_aspect("equal", 'box')
    ax.set_title(title)

fig, ax = plt.subplots(figsize=(5, 3))
add_mode(ax, dmd.modes[:, 0].real, "mode 0")
plt.show()

# plot reconstructed field at t=9 to 9.7s
fig, axarr = plt.subplots(4, 4, figsize=(8, 6), sharex=True, sharey=True)
count = 1
for row in range(4):
    add_mode(axarr[row, 0], dmd.modes[:, count].real, f"mode {count}, real")
    add_mode(axarr[row, 1], dmd.modes[:, count].imag, f"mode {count}, imag.")
    count += 2
    add_mode(axarr[row, 2], dmd.modes[:, count].real, f"mode {count}, real")
    add_mode(axarr[row, 3], dmd.modes[:, count].imag, f"mode {count}, imag.")
    count += 2
plt.show()


reconstruction = dmd.reconstruction

fig, axarr = plt.subplots(4, 4, figsize=(8, 6), sharex=True, sharey=True)
count = 0
for row in range(4):
    add_mode(axarr[row, 0], data_matrix[:, count], f"org., t={window_times[count]}s")
    add_mode(axarr[row, 1], reconstruction[:, count], f"reconstr., t={window_times[count]}s")
    count += 4
    add_mode(axarr[row, 2], data_matrix[:, count], f"org., t={window_times[count]}s")
    add_mode(axarr[row, 3], reconstruction[:, count], f"reconstr., t={window_times[count]}s")
    count += 4
plt.show()

# animate the reconstruction over time
from matplotlib.animation import FuncAnimation
from matplotlib import cm

# create a figure and axis object no subplot
fig, ax = plt.figure(figsize=(8, 6)), plt.axes()
ax.set_aspect("equal", 'box')
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
cmap = cm.get_cmap("jet")
vmin, vmax = reconstruction.min().item(), reconstruction.max().item()
sc = ax.tricontourf(x, y, reconstruction[:, 0], levels=15, cmap=cmap, vmin=vmin, vmax=vmax)
ax.tricontour(x, y, reconstruction[:, 0], levels=15, linewidths=0.1, colors='k')
ax.add_patch(plt.Circle((0.2, 0.2), 0.05, color='k'))

def update(i):
    for coll in ax.collections:
        coll.remove()
    sc = ax.tricontourf(x, y, reconstruction[:, i], levels=15, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.tricontour(x, y, reconstruction[:, i], levels=15, linewidths=0.1, colors='k')
    ax.set_title(f"t={window_times[i]}s")
    return sc

ani = FuncAnimation(fig, update, frames=range(reconstruction.shape[1]), interval=100)
plt.show()
ani.save("reconstruction.gif", writer="pillow", fps=10)

