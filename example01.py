import bisect
import torch as pt
import matplotlib.pyplot as plt
import matplotlib as mpl
from flowtorch import DATASETS
from flowtorch.data import FOAMDataloader, mask_box
from flowtorch.analysis import SVD

# increase resolution of plots
mpl.rcParams['figure.dpi'] = 160

# path to the dataset relative to the current working directory
path = "datasets_minimal/of_cylinder2D_binary"

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
data_matrix -= pt.mean(data_matrix, dim=1).unsqueeze(-1)

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

# Example usage
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

s = svd.s
s_sum = s.sum().item()
# relative contribution
s_rel = [s_i / s_sum * 100 for s_i in s]
# cumulative contribution
s_cum = [s[:n].sum().item() / s_sum * 100 for n in range(s.shape[0])]
# find out how many singular values we need to reach at least 99 percent
i_99 = bisect.bisect_right(s_cum, 99)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.bar(range(s.shape[0]), s_rel, align="edge")
ax2.plot(range(s.shape[0]), s_cum)
ax2.set_xlim(0, 90)
ax2.set_ylim(0, 105)
ax1.set_title("individual contribution in %")
ax2.set_title("cumulative contribution in %")
ax2.plot([0, i_99, i_99], [s_cum[i_99], s_cum[i_99], 0], ls="--", color="C3")
ax2.text(i_99+1, 45, "first {:d} singular values yield {:1.2f}%".format(i_99, s_cum[i_99]))
plt.show()



x = pt.masked_select(vertices[:, 0], mask)
y = pt.masked_select(vertices[:, 1], mask)

fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
count = 0
for row in range(2):
    for col in range(2):
        axarr[row, col].tricontourf(x, y, svd.U[:, count], levels=14, cmap="jet")
        axarr[row, col].tricontour(x, y, svd.U[:, count], levels=14, linewidths=0.5, colors='k')
        axarr[row, col].add_patch(plt.Circle((0.2, 0.2), 0.05, color='k'))
        axarr[row, col].set_aspect("equal", 'box')
        # add 1 for the POD mode number since we subtracted the mean
        axarr[row, col].set_title(f"mode {count + 1}")
        count += 1
plt.show()

times_num = [float(time) for time in window_times]

fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
count = 0
for row in range(2):
    for col in range(2):
        axarr[row, col].plot(times_num, svd.V[:, count]*svd.s[count], lw=1, label=f"coeff. mode {i+1}")
        axarr[row, col].set_xlim(min(times_num), max(times_num))
        axarr[row, col].grid()
        # add 1 for the POD mode number since we subtracted the mean
        axarr[row, col].set_title(f"mode coeff. {count + 1}")
        count += 1
for ax in axarr[1, :]:
    ax.set_xlabel(r"$t$ in $s$")
plt.show()

# plot the reconstructed field at t=4.0, 4,2, 4.4, 4.6