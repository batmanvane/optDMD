import torch as pt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogFormatterMathtext
from matplotlib.animation import FuncAnimation
from matplotlib import cm

from flowtorch.data import FOAMDataloader, mask_box
from flowtorch.analysis import SVD, DMD, OptDMD
from pydmd import CDMD

# Increase resolution of plots
mpl.rcParams['figure.dpi'] = 160

# Path to the dataset relative to the current working directory
#path = "datasets/of_cylinder2D_binary"
path = "datasets_minimal/of_cylinder2D_binary"

# Load dataset
loader = FOAMDataloader(path)
times = loader.write_times
fields = loader.field_names
print(f"Number of available snapshots: {len(times)}")
print("First five write times: ", times[:5])
print(f"Fields available at t={times[-1]}: ", fields[times[-1]])

# Extract 2D vertices and apply mask
vertices = loader.vertices[:, :2]
mask = mask_box(vertices, lower=[0.1, -1], upper=[0.75, 1])
print(f"Selected vertices: {mask.sum().item()}/{mask.shape[0]}")

# Define masked vertices for later use (e.g. in animations)
masked_vertices = vertices[mask]
x = masked_vertices[:, 0]
y = masked_vertices[:, 1]

# Plot the mask
every = 4  # Use only every 4th vertex
fig, ax = plt.subplots()
ax.scatter(vertices[::every, 0], vertices[::every, 1], s=0.5, c=mask[::every])
ax.set_aspect("equal", "box")
plt.show()

# Load the vorticity field at times >= 4.0
# Convert write times to float for arithmetic and comparison
window_times = [float(time) for time in times if float(time) >= 4.0]
dt = window_times[1] - window_times[0]

# Build the data matrix by selecting the z-component and applying the mask
data_matrix = pt.zeros((mask.sum().item(), len(window_times)), dtype=pt.float32)
def format_time(t: float) -> str:
    return str(int(t)) if t.is_integer() else str(t)

for i, time in enumerate(window_times):
    # Load the vorticity vector field, take the z-component [:, 2], and apply the mask
    formatted_time = format_time(time) # Convert float to string for loader 4.0 -> "4" 4.1 -> "4.1"
    snapshot = loader.load_snapshot("vorticity", formatted_time)
    data_matrix[:, i] = pt.masked_select(snapshot[:, 2], mask)

# subtract the temporal mean
data_matrix -= pt.mean(data_matrix, dim=1).unsqueeze(-1)

# numpy data matrix
data_matrix_np = data_matrix.cpu().numpy()

random_matrix = np.random.permutation(data_matrix_np.shape[0] * data_matrix_np.shape[1])
random_matrix = random_matrix.reshape(data_matrix_np.shape[1], data_matrix_np.shape[0])

compression_matrix = random_matrix / np.linalg.norm(random_matrix)

# Create flowtorch SVD with an automatic rank selection (here using rank=400 as a parameter)
svd = SVD(data_matrix, rank=400)
print(svd)

# Compute truncated SVD using PyTorch with an explicit rank
def compute_svd_torch(data_matrix, rank):
    """Compute truncated SVD using PyTorch and return U, S, Vh with specified rank."""
    U, S, Vh = pt.linalg.svd(data_matrix, full_matrices=False)
    return U[:, :rank], S[:rank], Vh[:rank, :]

rank_pt = 41
U_trunc, S_trunc, Vh_trunc = compute_svd_torch(data_matrix, rank_pt)
truncated_svd_pt_size_mb = (U_trunc.numel() * 4 + S_trunc.numel() * 4 + Vh_trunc.numel() * 4) / (1024**2)
print(f"Truncated SVD Size (PyTorch): {truncated_svd_pt_size_mb:.4f} MB")

# Functions to estimate SVD memory usage (using 4 bytes per element for float32)
def svd_full_size(data_matrix):
    m, n = data_matrix.shape
    r = min(m, n)  # Rank for full SVD
    size_U = m * m * 4      # U: m x m
    size_S = r * r * 4      # Sigma: r x r
    size_Vh = r * n * 4     # Vh: r x n
    return (size_U + size_S + size_Vh) / (1024 ** 2)

def svd_truncated_size(data_matrix, rank):
    m, n = data_matrix.shape
    size_U = m * rank * 4   # U: m x rank
    size_S = rank * rank * 4  # Sigma: rank x rank
    size_Vh = rank * n * 4   # Vh: rank x n
    return (size_U + size_S + size_Vh) / (1024 ** 2)

full_svd_size_mb = svd_full_size(data_matrix)
print(f"Estimated size of full SVD: {full_svd_size_mb:.4f} MB")

truncated_svd_size_mb = svd.required_memory / (1024 ** 2)  # from flowtorch SVD instance

def compression_rate(full_size, truncated_size):
    """Compute compression rate and compression ratio."""
    rate = truncated_size / full_size  # Lower is better
    ratio = full_size / truncated_size   # Higher is better
    return rate, ratio

rate, ratio = compression_rate(full_svd_size_mb, truncated_svd_size_mb)
print(f"Full SVD Size: {full_svd_size_mb:.4f} MB")
print(f"Truncated SVD Size: {truncated_svd_size_mb:.4f} MB")
print(f"Compression Rate: {rate:.4f} (lower is better)")
print(f"Compression Ratio: {ratio:.2f}x (higher is better)")

# Reconstruct the field starting from t=4.0 and compute the reconstruction error
print(f"Rank: {svd.opt_rank:.0f}")
reconstructed_data = svd.reconstruct(svd.opt_rank)
mse = pt.mean(pt.abs(data_matrix - reconstructed_data) ** 2)
print(f"Mean squared error: {mse:.3f}")

# ----------------------------------------------------------------------
# Compute errors and compression metrics for a range of ranks
ranks = list(range(1, svd.rank + 1))
errors = []
errors_DMD = []
errors_OptDMD = []
errors_CDMD = []
compression_rates = []
compression_ratios = []

for r in ranks:
    # SVD reconstruction error
    try:
        svd_r = SVD(data_matrix, rank=r)
        rec_svd = svd_r.reconstruct(r)
        mse_svd = pt.linalg.norm(data_matrix - rec_svd, dim=0)
        errors.append(mse_svd.mean().item())
    except Exception as e:
        print(f"Warning: SVD reconstruction failed for rank {r} with error: {e}")
        errors.append(np.nan)

    # DMD reconstruction error
    try:
        dmd_r = DMD(data_matrix, dt=dt, rank=r)
#        dmd_r = DMD(data_matrix, dt=dt, rank=r, robust={"sparsity": 10.0, "verbose": False, "max_iter": 100})
        rec_dmd = dmd_r.reconstruction
        mse_dmd = pt.linalg.norm(data_matrix - rec_dmd, dim=0)
        errors_DMD.append(mse_dmd.mean().item())
    except Exception as e:
        print(f"Warning: DMD reconstruction failed for rank {r} with error: {e}")
        errors_DMD.append(np.nan)

    # OptDMD reconstruction error
    try:
        optdmd_r = OptDMD(data_matrix, dt=dt, rank=r)
        optdmd_r.train(train_size=0.9, val_size=0.1, lr=1e-4, stopping_options={"patience": 80, "min_delta": 5e-6})
        rec_optdmd = optdmd_r.reconstruction
        mse_optdmd = pt.linalg.norm(data_matrix - rec_optdmd, dim=0)
        errors_OptDMD.append(mse_optdmd.mean().item())

        # fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        # t = pt.linspace(0, 2 * np.pi, 100)
        # ax.plot(pt.cos(t), pt.sin(t), ls="--", color="k", lw=2)
        # ax.scatter(optdmd_r.eigvals.real, optdmd_r.eigvals.imag, zorder=7)
        # ax.set_xlim(-1.3, 1.3)
        # ax.set_ylim(-1.3, 1.3)
        # ax.set_xlabel(r"$Re(\lambda)$")
        # ax.set_ylabel(r"$Im(\lambda)$")
        # for i, val in enumerate(optdmd_r
        #                                 .eigvals):
        #     index = "{" + f"{i}" + "}"
        #     ax.annotate(r"$\lambda_{:s}$".format(index), (val.real * 1.13, val.imag * 1.13), ha='center', va="center")
        # plt.show()
        # pause = input("Press Enter to continue...")

    except Exception as e:
        print(f"Warning: OPTDMD reconstruction failed for rank {r} with error: {e}")
        errors_OptDMD.append(np.nan)

    # CDMD reconstruction error
    try:
        cdmd = CDMD(svd_rank=r, compression_matrix="uniform")
        cdmd.fit(data_matrix_np)
        rec_cdmd = cdmd.reconstructed_data
        mse_cdmd = np.linalg.norm(data_matrix_np - rec_cdmd)
        errors_CDMD.append(mse_cdmd.mean().item())
    except Exception as e:
        print(f"Warning: CDMD reconstruction failed for rank {r} with error: {e}")
        errors_CDMD.append(np.nan)

    # Compression metrics using our own memory estimates
    truncated_size = svd_truncated_size(data_matrix, r)
    rate_r, ratio_r = compression_rate(full_svd_size_mb, truncated_size)
    compression_rates.append(rate_r)
    compression_ratios.append(ratio_r)

# Plot Rank vs. MSE (SVD vs. DMD vs. OptDMD vs. CDMD) and Compression Ratio
fig, ax1 = plt.subplots()
ax1.plot(ranks, errors, label="MSE (SVD)", color="C0", linestyle="--")
ax1.plot(ranks, errors_DMD, label="MSE (DMD)", color="C2", linestyle=":")
ax1.plot(ranks, errors_OptDMD, label="MSE (OptDMD)", color="C3", linestyle="-.")
ax1.plot(ranks, errors_CDMD, label="MSE (CDMD)", color="C4", linestyle="-")
ax1.set_xlabel("Rank")
ax1.set_ylabel("Mean Squared Error")
ax1.set_title("Compression Rate vs Mean Squared Error")
ax1.grid()
ax1.legend()

ax2 = ax1.twinx()
ax2.plot(ranks, compression_ratios, label="Compression Ratio", color="C1")
ax2.set_ylabel("Compression Ratio")
plt.show()

# Plot MSE vs. Compression Ratio on log-log axes
x_ticks = np.logspace(np.floor(np.log10(min(compression_ratios))),
                      np.ceil(np.log10(max(compression_ratios))),
                      num=5)
fig, ax1 = plt.subplots()
ax1.plot(compression_ratios, errors, marker="o", linestyle="--", color="C0", label="MSE (SVD)")
ax1.plot(compression_ratios, errors_DMD, marker="s", linestyle=":", color="C2", label="MSE (DMD)")
ax1.plot(compression_ratios, errors_OptDMD, marker="s", linestyle="-.", color="C3", label="MSE (OptDMD)")
ax1.plot(compression_ratios, errors_CDMD, marker="s", linestyle="-", color="C4", label="MSE (CDMD)")
ax1.set_xlabel("Compression Ratio")
ax1.set_ylabel("Mean Squared Error")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xticks(x_ticks)
ax1.xaxis.set_major_formatter(LogFormatterMathtext())
ax1.grid(which="both", linestyle="--", linewidth=0.5)
ax1.set_title("Compression Ratio & Rank vs Mean Squared Error")
ax1.legend()

# Create a twin x-axis for Rank labels
ax2 = ax1.twiny()
ax2.set_xscale("log")
ax2.set_xlim(ax1.get_xlim())
# Select indices to label certain ranks
selected_indices = sorted(set([0, len(ranks) - 1] + [i for i, r in enumerate(ranks) if r in {1, 5, 10, 15, 20, 30}]))
selected_ranks = [ranks[i] for i in selected_indices]
selected_compression_ratios = [compression_ratios[i] for i in selected_indices]
ax2.set_xticks(selected_compression_ratios)
ax2.set_xticklabels([str(r) for r in selected_ranks])
ax2.set_xlabel("Rank")
plt.show()

# ----------------------------------------------------------------------
# Create Animated GIFs for SVD and DMD reconstructions
#
# def create_animation(rank, filename, method):
#     """Create and save an animation for SVD or DMD reconstruction."""
#     if method == "SVD":
#         reconstruction = svd.reconstruct(rank)
#     elif method == "DMD":
#         # Compute a new DMD instance for the given rank
#         dmd_inst = DMD(data_matrix, dt=dt, rank=rank)
#         reconstruction = dmd_inst.reconstruction
#     else:
#         raise ValueError("Method must be either 'SVD' or 'DMD'.")
#
#     # Find the start index corresponding to time >= 4.0 (times are already floats)
#     start_index = next(i for i, t in enumerate(window_times) if t >= 4.0)
#     reconstruction = reconstruction[:, start_index:]
#     anim_times = window_times[start_index:]
#
#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.set_aspect("equal", "box")
#     ax.set_xlabel(r"$x$")
#     ax.set_ylabel(r"$y$")
#
#     cmap = cm.get_cmap("jet")
#     vmin, vmax = reconstruction.min().item(), reconstruction.max().item()
#
#     # Initial plot using tricontourf (automatically triangulates x, y)
#     sc = ax.tricontourf(x, y, reconstruction[:, 0], levels=15, cmap=cmap, vmin=vmin, vmax=vmax)
#     ax.tricontour(x, y, reconstruction[:, 0], levels=15, linewidths=0.1, colors='k')
#     ax.add_patch(plt.Circle((0.2, 0.2), 0.05, color='k'))
#
#     def update(i):
#         for coll in ax.collections:
#             coll.remove()
#         sc = ax.tricontourf(x, y, reconstruction[:, i], levels=15, cmap=cmap, vmin=vmin, vmax=vmax)
#         ax.tricontour(x, y, reconstruction[:, i], levels=15, linewidths=0.1, colors='k')
#         ax.set_title(f"t={anim_times[i]:.1f}s ({method} Rank={rank})")
#         return sc
#
#     ani = FuncAnimation(fig, update, frames=len(anim_times), interval=100)
#     ani.save(filename, writer="pillow", fps=10)
#     plt.close(fig)
#
# # Generate GIFs for SVD and DMD at different ranks
# for method in ["SVD", "DMD"]:
#     create_animation(svd.rank, f"reconstruction_{method}_rank.gif", method)
#     create_animation(svd.opt_rank, f"reconstruction_{method}_opt_rank.gif", method)
#     create_animation(3, f"reconstruction_{method}_3.gif", method)
#
# print("GIFs saved for SVD and DMD reconstructions.")
