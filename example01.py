import bisect
import torch as pt
import matplotlib.pyplot as plt
import matplotlib as mpl
from flowtorch import DATASETS
from flowtorch.data import FOAMDataloader, mask_box
from flowtorch.analysis import SVD

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext

from flowtorch.analysis import DMD, OptDMD


# increase resolution of plots
mpl.rcParams['figure.dpi'] = 160

# path to the dataset relative to the current working directory
#path = "datasets_minimal/of_cylinder2D_binary"
path = "datasets/of_cylinder2D_binary"

loader = FOAMDataloader(path)
times = loader.write_times
fields = loader.field_names
print(f"Number of available snapshots: {len(times)}")
print("First five write times: ", times[:5])
print(f"Fields available at t={times[-1]}: ", fields[times[-1]])

vertices = loader.vertices[:, :2]
mask = mask_box(vertices, lower=[0.1, -1], upper=[0.75, 1])
print(f"Selected vertices: {mask.sum().item()}/{mask.shape[0]}")

# plot the mask
every = 4 # use only every 4th vertex
fig, ax = plt.subplots()
ax.scatter(vertices[::every, 0], vertices[::every, 1], s=0.5, c=mask[::every])
ax.set_aspect("equal", 'box')
#ax.set_xlim(0.0, 2.2)
#ax.set_ylim(0.0, 0.41)
plt.show()

# load the vorticity field at times >= 4.0
# extract mesh vertices and apply the mask
# drop z-component of the vorticity vector field (simulation is 2D)
window_times = [time for time in times if float(time) >= 4.0]
data_matrix = pt.zeros((mask.sum().item(), len(window_times)), dtype=pt.float32)
for i, time in enumerate(window_times):
    # load the vorticity vector field, take the z-component [:, 2], and apply the mask
    data_matrix[:, i] = pt.masked_select(loader.load_snapshot("vorticity", time)[:, 2], mask)

# subtract the temporal mean
# data_matrix -= pt.mean(data_matrix, dim=1).unsqueeze(-1)

#flowtorch svd with automatic rank selection --> check criterion
svd = SVD(data_matrix, rank=400)
print(svd)

#torch svd with explicit rank selection

def compute_svd_torch(data_matrix, rank):
    """Compute truncated SVD using PyTorch and return U, S, Vh with specified rank."""
    U, S, Vh = pt.linalg.svd(data_matrix, full_matrices=False)  # Faster economy SVD
    return U[:, :rank], S[:rank], Vh[:rank, :]

#rank = svd.opt_rank
rank_pt = 41
U_trunc, S_trunc, Vh_trunc = compute_svd_torch(data_matrix, rank_pt)

# Compute memory usage
truncated_svd_pt_size_mb = (U_trunc.numel() * 4 + S_trunc.numel() * 4 + Vh_trunc.numel() * 4) / (1024**2)
print(f"Truncated SVD Size (PyTorch): {truncated_svd_pt_size_mb:.4f} MB")


### DMD
dt = float(window_times[1]) - float(window_times[0])
#dmd = DMD(data_matrix, dt=dt, rank=svd.rank)
#dmdopt = OptDMD(data_matrix, dt=dt, rank=svd.opt_rank)
#print(dmd)


def svd_full_size(data_matrix):
    m, n = data_matrix.shape
    r = min(m, n)  # Rank for full SVD

    # Storage in bytes:
    # U: m x m (complex64 -> 8 bytes per element)
    size_U = m * m * 8
    # Sigma: r x r (float32 -> 4 bytes per element)
    size_S = r * r * 4
    # Vh: r x n (complex64 -> 8 bytes per element)
    size_Vh = r * n * 8

    total_size_bytes = size_U + size_S + size_Vh
    total_size_mb = total_size_bytes / (1024 ** 2)  # Convert to MB

    return total_size_mb
def svd_truncated_size(data_matrix, rank):
    m, n = data_matrix.shape

    # Storage in bytes:
    # U: m x rank (complex64 -> 8 bytes per element)
    size_U = m * rank * 8
    # Sigma: rank x rank (float32 -> 4 bytes per element)
    size_S = rank * rank * 4
    # Vh: rank x n (complex64 -> 8 bytes per element)
    size_Vh = rank * n * 8

    total_size_bytes = size_U + size_S + size_Vh
    total_size_mb = total_size_bytes / (1024 ** 2)  # Convert to MB

    return total_size_mb

# Print estimated size of full SVD
full_svd_size_mb = svd_full_size(data_matrix)
print(f"Estimated size of full SVD: {full_svd_size_mb:.4f} MB")


truncated_svd_size_mb=svd.required_memory/ (1024 ** 2) # Convert to MB

def compression_rate(full_size, truncated_size):
    """Compute compression rate and compression ratio."""
    rate = truncated_size / full_size  # Compression rate (lower is better)
    ratio = full_size / truncated_size  # Compression ratio (higher is better)
    return rate, ratio

rate, ratio = compression_rate(full_svd_size_mb, truncated_svd_size_mb)

print(f"Full SVD Size: {full_svd_size_mb:.4f} MB")
print(f"Truncated SVD Size: {truncated_svd_size_mb:.4f} MB")
print(f"Compression Rate: {rate:.4f} (lower is better)")
print(f"Compression Ratio: {ratio:.2f}x (higher is better)")


# reconstruct the field starting from t=4.0
print(f"Rank: {svd.opt_rank:.0f}")
reconstructed_data = svd.reconstruct(svd.opt_rank)

# calculate the mean squared error to assess the quality of the reconstruction
mse = pt.mean(pt.abs(data_matrix - reconstructed_data) ** 2)
print(f"mean squared error: {mse:.3f}")


#plot compression rate vs mse via screening the rank from 1 to 50
ranks = range(1, svd.rank+1)
errors = []
errors_DMD = []
compression_rates = []
compression_ratio = []
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext
from matplotlib.animation import FuncAnimation
from matplotlib import cm

# Compute errors and compression metrics
ranks = list(range(1, svd.rank + 1))
errors = []
errors_DMD = []
errors_OptDMD = []
compression_rates = []
compression_ratio = []

for r in ranks:
    # Compute Mean Squared Error (MSE) for SVD
    #errors.append(pt.mean(pt.abs(data_matrix - svd.reconstruct(r)) ** 2).item())

    try:
        reconstructed_svd = SVD(data_matrix, rank=r)
        mse_svd = pt.linalg.norm(data_matrix - reconstructed_svd.reconstruct(r), dim=0)
        errors.append(mse_svd.mean().item())
    except Exception as e:
        print(f"Warning: SVD reconstruction failed for rank {r} with error: {e}")
        errors.append(np.nan)


    # Compute Mean Squared Error (MSE) for DMD
    try:
       # reconstructed_dmd = dmd.partial_reconstruction(set(range(1, r + 1)))
        reconstructed_dmd = DMD(data_matrix, dt=dt, rank=r)
        mse_dmd = pt.linalg.norm(data_matrix - reconstructed_dmd.reconstruction, dim=0)
#        mse_dmd = pt.mean(pt.abs(data_matrix - reconstructed_dmd.reconstruction) ** 2)
        errors_DMD.append(mse_dmd.mean().item())

 #       errors_DMD.append(reconstructed_dmd.reconstruction_error.mean().item())
    except Exception as e:
        print(f"Warning: DMD reconstruction failed for rank {r} with error: {e}")
        errors_DMD.append(np.nan)  # Assign NaN if DMD reconstruction fails

    # Compute Mean Squared Error (MSE) for OPTDMD
    try:
        reconstructed_optdmd = OptDMD(data_matrix, dt=dt, rank=r)
        mse_optdmd = pt.linalg.norm(data_matrix - reconstructed_optdmd.reconstruction, dim=0)
        errors_OptDMD.append(mse_optdmd.mean().item())
    except Exception as e:
        print(f"Warning: OPTDMD reconstruction failed for rank {r} with error: {e}")
        errors_OptDMD.append(np.nan)


    # Calculate size of truncated SVD for rank r
    truncated_svd_size_mb = svd_truncated_size(data_matrix, r)
    rate, ratio = compression_rate(full_svd_size_mb, truncated_svd_size_mb)
    compression_ratio.append(ratio)
    compression_rates.append(rate)

# ** Plot Rank vs. MSE (SVD vs. DMD) and Compression Ratio **
fig, ax1 = plt.subplots()

ax1.plot(ranks, errors, label="MSE (SVD)", color="C0", linestyle="--")
ax1.plot(ranks, errors_DMD, label="MSE (DMD)", color="C2", linestyle=":")
ax1.plot(ranks, errors_OptDMD, label="MSE (OptDMD)", color="C3", linestyle="-.")
ax1.set_xlabel("Rank")
ax1.set_ylabel("Mean Squared Error")
ax1.set_title("Compression Rate vs Mean Squared Error")
ax1.grid()
ax1.legend()

# Twin axis for compression ratio
ax2 = ax1.twinx()
ax2.plot(ranks, compression_ratio, label="Compression Ratio", color="C1")
ax2.set_ylabel("Compression Ratio")

plt.show()

# ** Plot MSE vs. Compression Ratio (SVD vs. DMD) **
selected_indices = [i for i, r in enumerate(ranks) if r in {1, 5, 10, 15, 20, 30}]
selected_indices += [0, len(ranks) - 1]  # Ensure lowest and highest ranks are included
selected_indices = sorted(set(selected_indices))

selected_ranks = [ranks[i] for i in selected_indices]
selected_compression_ratios = [compression_ratio[i] for i in selected_indices]

# Define reasonable tick positions for Compression Ratio
x_ticks = np.logspace(np.floor(np.log10(min(compression_ratio))),
                      np.ceil(np.log10(max(compression_ratio))),
                      num=5)  # Ensure multiple ticks in log scale

fig, ax1 = plt.subplots()

# Plot Mean Squared Error vs. Compression Ratio
ax1.plot(compression_ratio, errors, marker="o", linestyle="--", color="C0", label="MSE (SVD)")
ax1.plot(compression_ratio, errors_DMD, marker="s", linestyle=":", color="C2", label="MSE (DMD)")
ax1.plot(compression_ratio, errors_OptDMD, marker="s", linestyle="-.", color="C3", label="MSE (OptDMD)")
ax1.set_xlabel("Compression Ratio")
ax1.set_ylabel("Mean Squared Error")

# Apply explicit tick settings
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xticks(x_ticks)
ax1.xaxis.set_major_formatter(LogFormatterMathtext())  # Use 10³ format

ax1.tick_params(axis='y', labelcolor="C0")
ax1.grid(which="both", linestyle="--", linewidth=0.5)
ax1.set_title("Compression Ratio & Rank vs Mean Squared Error")
ax1.legend()

# Create a twin x-axis for Rank
ax2 = ax1.twiny()
ax2.set_xscale("log")
ax2.set_xlim(ax1.get_xlim())  # Ensure both x-axes are aligned
ax2.set_xticks(selected_compression_ratios)
ax2.set_xticklabels([str(r) for r in selected_ranks])
ax2.set_xlabel("Rank")

plt.show()


# ** Create Animated GIFs for SVD and DMD Reconstructions **
def create_animation(rank, filename, method):
    """ Create and save the animation for SVD or DMD """
    reconstruction = svd.reconstruct(rank) if method == "SVD" else dmd.reconstruct(rank)

    # Find the start index corresponding to 4s
    start_index = next(i for i, t in enumerate(window_times) if t >= 4)

    # Trim the reconstruction and time arrays
    reconstruction = reconstruction[:, start_index:]
    times = window_times[start_index:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal", 'box')
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    cmap = cm.get_cmap("jet")
    vmin, vmax = reconstruction.min().item(), reconstruction.max().item()

    # Initial plot
    sc = ax.tricontourf(x, y, reconstruction[:, 0], levels=15, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.tricontour(x, y, reconstruction[:, 0], levels=15, linewidths=0.1, colors='k')
    ax.add_patch(plt.Circle((0.2, 0.2), 0.05, color='k'))

    def update(i):
        """ Update function for the animation """
        for coll in ax.collections:
            coll.remove()
        sc = ax.tricontourf(x, y, reconstruction[:, i], levels=15, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.tricontour(x, y, reconstruction[:, i], levels=15, linewidths=0.1, colors='k')
        ax.set_title(f"t={times[i]:.1f}s ({method} Rank={rank})")
        return sc

    ani = FuncAnimation(fig, update, frames=len(times), interval=100)

    # Save the animation as a GIF
    ani.save(filename, writer="pillow", fps=10)
    plt.close(fig)


# Generate GIFs for SVD and DMD at three ranks
for method in ["SVD", "DMD"]:
    create_animation(svd.rank, f"reconstruction_{method}_rank.gif", method)
    create_animation(svd.opt_rank, f"reconstruction_{method}_opt_rank.gif", method)
    create_animation(3, f"reconstruction_{method}_3.gif", method)

print(
    "GIFs saved for SVD and DMD: reconstruction_SVD_rank.gif, reconstruction_SVD_opt_rank.gif, reconstruction_SVD_3.gif")
print(
    "GIFs saved for SVD and DMD: reconstruction_DMD_rank.gif, reconstruction_DMD_opt_rank.gif, reconstruction_DMD_3.gif")
