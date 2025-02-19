
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch as pt

# increase resolution of plots
mpl.rcParams['figure.dpi'] = 160

# create some data
x = pt.linspace(-10, 10, 50)
t = pt.linspace(0, 6*np.pi, 40)
X, T = pt.meshgrid(x, t)

def plot_as_line(data, x, t, every, show_imag=False):
    fig, ax = plt.subplots(figsize=(8, 3))
    t_selected = t[::every]
    data_selected = data[:, ::every]
    opacity_factor = 0.5
    for i in range(len(t_selected)):
        if i == 0:
            ax.plot(x, data[:, i].real, c="C0", alpha=1/(1+i*opacity_factor), ls="-", label=r"$Re(f)$")
            if show_imag:
                ax.plot(x, data[:, i].imag, c="C1", alpha=1/(1+i*opacity_factor), ls="--", label=r"$Im(f)$")
        else:
            ax.plot(x, data[:, i].real, c="C0", alpha=1/(1+i*opacity_factor), ls="-")
            if show_imag:
                ax.plot(x, data[:, i].imag, c="C1", alpha=1/(1+i*opacity_factor), ls="--")
    ax.set_xlim(-10, 10)
    ax.set_xlabel(r"$x$")
    ax.legend()
    ax.set_title("time progression indicated by decreasing opacity")

# two oscillating hyperbolic functions
# The positive real part in the exponent gives the oscillation a slowly growing amplitude.

f1 = pt.multiply(10.0*pt.tanh(X/2.0)/pt.cosh(X/2.0), pt.exp((0.1+2.8j)*T))
plot_as_line(f1, x, t, 4, True)
plt.show()

# oscillating parabola
# The negative real part in the exponent gives the oscillation a slowly decaying amplitude.

f2 = pt.multiply(20-0.2*pt.pow(X, 2), pt.exp((-0.05+2.3j)*T))
plot_as_line(f2, x, t, 4, True)
plt.show()

# oscillating line (like a seesaw)
# The real part in the exponent is zero, so the amplitude stays constant.
# In comparison to the other two functions, f3 oscillates at the lowest frequency.

f3 = pt.multiply(X, pt.exp(0.6j*T))
plot_as_line(f3, x, t, 4, True)
plt.show()

# generate data based on superposition of the three functions
data = f1 + f2 + f3
plot_as_line(data, x, t, 4)
plt.show()

def show_as_image(data, X, T):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    ax1.pcolormesh(T.T, X.T, data.T.real, shading="nearest")
    ax2.pcolormesh(T.T, X.T, data.T.imag, shading="nearest")
    ax1.set_xlabel(r"$t$")
    ax2.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$x$")
    ax1.set_title(r"$Re(f)$")
    ax2.set_title(r"$Im(f)$")

show_as_image(data, X, T)
plt.show()

# add gaussian noise to the data
data_noisy = data + pt.normal(.02*pt.ones_like(X), 2*pt.ones_like(X))
show_as_image(data_noisy, X, T)
plt.show()


# SVD of the data
U, s, VH = pt.linalg.svd(data[:, :-1])
Un, sn, VHn = pt.linalg.svd(data_noisy[:, :-1])

# plot singular values
# In the original dataset, three non-zero singular values are identified.
# Adding noise yields many more singular values with smaller but non-zero values.
# Also the largest singular value differs slightly from that of the clean data.

fig, ax = plt.subplots(figsize=(8, 3))
ax.bar(range(s.size()[0]), s, color="C0", label="data")
ax.bar(range(sn.size()[0]), sn, ec="C1", fill=False, label="noisy data")
ax.set_xlabel(r"$i$")
ax.set_ylabel(r"$\sigma_i$")
ax.set_xlim(-1, 30)
ax.legend()
plt.show()


# Truncated SVD
# The first three singular values are used to reconstruct the data.
rank = 3
Ur, sr, Vr = U[:, :rank], s[:rank], VH.conj().T[:, :rank]
Unr, snr, Vnr = Un[:, :rank], sn[:rank], VHn.conj().T[:, :rank]

# Next, we compute the reduced linear operator
# using the time-shifted matrix and the truncated SVD.

At = Ur.conj().T @ data[:, 1:] @ Vr @ pt.diag(1.0/sr).type(pt.cfloat)
Atn = Unr.conj().T @ data[:, 1:] @ Vnr @ pt.diag(1.0/snr).type(pt.cfloat)


# The real and imaginary parts of the operator’s eigenvalues tell us about the time-dynamics of individual modes.
# First, we plot the eigenvalues in the imaginary plane together with a unit-circle.
# Note that for the time-discrete system, the modes are:
# - stable if the eigenvalues are on the circle
# - growing (unstable) if the eigenvalues are outside the circle
# - shrinking (stable) if the eigenvalues are inside the circle
# As can be seen in the image below, the dynamics of all three components are identified correctly.
# The noise introduces some error. However, thanks to the truncation, we already removed a significant portion of the noise,
# and the identified dynamics remain close to the ground truth.

val, vec = pt.linalg.eig(At)
valn, vecn = pt.linalg.eig(Atn)

p = pt.linspace(0, np.pi/2, 100)

fig, ax = plt.subplots()
ax.plot(pt.cos(p), pt.sin(p), c="k")
ax.fill_between(pt.cos(p),0, pt.sin(p), color="k", alpha=0.2)
ax.scatter(val.real, val.imag, label="data", zorder=6)
ax.scatter(valn.real, valn.imag, label="noisy data", zorder=6)
for i, ev in enumerate(val):
    ax.text(ev.real+0.02, ev.imag+0.02, r"$\lambda_{:d}$".format(i))
ax.text(0.2, 0.2, "shrinking modes", rotation=45)
ax.text(0.75, 0.75, "growing modes", rotation=45)
ax.set_aspect("equal")
ax.set_xlim(0, 1.1)
ax.set_ylim(0, 1.1)
ax.set_xlabel(r"$Re(\lambda_i)$")
ax.set_ylabel(r"$Im(\lambda_i)$")
ax.legend(loc=8, ncol=2)
plt.show()


# Let’s now compute the eigenvalues of the corresponding continuous linear operator.
# From the real and imaginary parts of all, we can directly read off growth rates and frequencies.

dt = t[1] - t[0]
lambda_t = pt.log(val) / dt
lambdan_t = pt.log(valn) / dt
print("clean dataset")
for i, l in enumerate(lambda_t):
    print(r"Mode {:d} has a growth rate of {:+2.3f} and oscillates with a frequency of {:2.3f}/(2π)Hz.".format(i, l.real, l.imag))
print("\nnoisy dataset")
for i, l in enumerate(lambdan_t):
    print(r"Mode {:d} has a growth rate of {:+2.3f} and oscillates with a frequency of {:2.3f}/(2π)Hz.".format(i, l.real, l.imag))


# In the final part of this notebook, we reconstruct the eigenvectors of the full linear operator, the so-called DMD modes,
# and visualize the difference between projected and exact DMD modes.
# The DMD modes are the eigenvectors of the linear operator and describe the spatial structure of the modes.

phi = data[:, 1:] @ Vr @ pt.diag(1.0/sr).type(pt.cfloat) @ vec
phin = data_noisy[:, 1:] @ Vnr @ pt.diag(1.0/snr).type(pt.cfloat) @ vecn
phi_pro = Ur @ vec

fig, ax = plt.subplots(figsize=(8, 3))
colors = ["C{:1d}".format(i) for i in range(3)]
for i in range(rank):
    ax.plot(x, phi[:, i].real, c=colors[i], ls="-", label=r"$\varphi_{:d}$".format(i))
    ax.plot(x, phin[:, i].real, c=colors[i], ls="--", label=r"$\varphi_{:d}$, noisy".format(i))
    ax.plot(x, phi_pro[:, i].real, c=colors[i], ls="-.", label=r"$\tilde{\varphi}" + r"_{:d}$".format(i))
ax.set_xlim(-10, 10)
ax.legend(bbox_to_anchor=(0.5, 1.0), loc=8, ncol=3)
plt.show()


# calculate the relative error of the DMD modes
# The relative error is calculated as the Frobenius norm of the difference between the exact and projected DMD modes
# divided by the Frobenius norm of the exact DMD modes.

def relative_error(phi, phi_pro):
    return pt.norm(phi - phi_pro) / pt.norm(phi)

errors = [relative_error(phi[:, i], phi_pro[:, i]) for i in range(rank)]
print("relative error of DMD modes:")
for i, e in enumerate(errors):
    print(f"mode {i}: {e.item():.3f}")

# # Ensure phi and phi_pro exist before running computations
# try:
#     # Compute norms
#     phi_norm = pt.norm(phi)
#     phi_pro_norm = pt.norm(phi_pro)
#     difference_norm = pt.norm(phi - phi_pro)
#
#     # Compute relative errors per mode
#     relative_error_values = [relative_error(phi[:, i], phi_pro[:, i]).item() for i in range(rank)]
#
#     # Store results in dictionary
#     norms_info = {
#         "||phi||": phi_norm.item(),
#         "||phi_pro||": phi_pro_norm.item(),
#         "||phi - phi_pro||": difference_norm.item(),
#         "Relative Errors": relative_error_values,
#     }
#
# except NameError as e:
#     norms_info = {"Error": str(e)}
#
# print(norms_info)

# calculate mean squared error on data and reconstructed data
data_reconstructed = Ur @ pt.diag(sr).type(pt.cfloat) @ Vr.conj().T
#mse = pt.mean(pt.square(data - data_reconstructed))

#mse = pt.mean(pt.square(data[:, :-1] - data_reconstructed))
mse = pt.mean(pt.abs(data[:, :-1] - data_reconstructed) ** 2)

print(f"mean squared error: {mse:.3f}")

#def compression_rate(data, Ur, sr, Vr):
#    return (Ur.size()[0] + sr.size()[0] + Vr.size()[0]) / data.size()[0]

def compression_rate(data, Ur, sr, Vr):
    """
    Calculate the compression rate for DMD decomposition.

    Parameters:
    data : array-like
        The original data matrix (e.g., size n_x x n_t).
    Ur : array-like
        The left singular vectors (spatial modes) of size (n_x x r).
    sr : array-like
        The singular values (vector or diagonal matrix) of size (r).
    Vr : array-like
        The right singular vectors (temporal coefficients) of size (n_t x r).

    Returns:
    float
        Compression rate, defined as the ratio of the original data size
        to the compressed data size. A higher value indicates better compression.

    Formula:
        Compression Rate = Original Size / Compressed Size
        Original Size = data.numel()
        Compressed Size = Ur.numel() + sr.numel() + Vr.numel()
    """
    original_size = data.numel()
    compressed_size = Ur.numel() + sr.numel() + Vr.numel()

    return original_size / compressed_size

# calculate the compression rate
compression = compression_rate(data, Ur, sr, Vr)
print(f"compression rate: {compression:.3f}")