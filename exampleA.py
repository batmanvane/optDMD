import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch as pt

# Increase resolution of plots
mpl.rcParams['figure.dpi'] = 160

class DynamicModeDecomposition:
    def __init__(self, data, rank):
        self.data = data
        self.rank = rank
        self.U, self.s, self.VH = pt.linalg.svd(data[:, :-1])
        self.Ur, self.sr, self.Vr = self.U[:, :rank], self.s[:rank], self.VH.conj().T[:, :rank]
        self.At = self.Ur.conj().T @ data[:, 1:] @ self.Vr @ pt.diag(1.0/self.sr).type(pt.cfloat)
        self.val, self.vec = pt.linalg.eig(self.At)
        self.phi = data[:, 1:] @ self.Vr @ pt.diag(1.0/self.sr).type(pt.cfloat) @ self.vec
        self.phi_pro = self.Ur @ self.vec

    def relative_error(self):
        """
        Calculate the relative error between the exact DMD modes (self.phi) and the projected DMD modes (self.phi_pro).

        The relative error is computed for each mode (up to the truncation rank) as the Frobenius norm of the difference
        between the exact and projected modes, normalized by the Frobenius norm of the exact modes.

        Returns:
            list: A list of relative errors for each mode, where each error corresponds to the relative difference
                  between the exact and projected DMD modes for that mode.
        """
        errors = [pt.norm(self.phi[:, i] - self.phi_pro[:, i]) / pt.norm(self.phi[:, i]) for i in range(self.rank)]
        return errors

    def compression_rate(self):
        """
        Calculate the compression rate achieved by the DMD process.

        The compression rate is defined as the ratio of the size of the original data
        to the size of the compressed representation (truncated SVD components).

        Returns:
            float: The compression rate, where a higher value indicates better compression.
        """
        original_size = self.data.numel()
        compressed_size = self.Ur.numel() + self.sr.numel() + self.Vr.numel()
        return original_size / compressed_size

    def reconstruct_data(self, t):
        """
        Reconstruct the data using DMD modes and eigenvalues.

        Parameters:
            t (torch.Tensor): The time vector (1D tensor of shape (n_t,)).

        Returns:
            torch.Tensor: The reconstructed data of shape (n_x, n_t).
        """
        # Compute the initial amplitudes of the modes
        b = pt.linalg.pinv(self.phi) @ self.data[:, 0]  # shape (rank,)

        # Compute the time dynamics matrix
        time_dynamics = pt.stack([self.val ** ti for ti in t], dim=1)  # shape (rank, n_t)

        # Reconstruct the data
        data_DMD = self.phi @ (b.unsqueeze(1) * time_dynamics)  # shape (n_x, n_t)
        return data_DMD


def reconstruction_quality(data, data_DMD):
    """
    Measure the reconstruction quality of DMD by comparing the original data (data)
    with the reconstructed data (data_DMD).

    Parameters:
        data (torch.Tensor): The original data matrix.
        data_DMD (torch.Tensor): The reconstructed data matrix.

    Returns:
        dict: A dictionary containing the relative error and MSE.
    """
    # Compute the relative error
    relative_error = pt.norm(data - data_DMD) / pt.norm(data)

    # Compute the mean squared error (MSE)
    mse = pt.mean((data - data_DMD)**2)

    return {
        "relative_error": relative_error.item(),
        "mse": mse.item()
    }

# Create some data
x = pt.linspace(-10, 10, 50)
t = pt.linspace(0, 60*np.pi, 40)
X, T = pt.meshgrid(x, t)

# Two oscillating hyperbolic functions
f1 = pt.multiply(10.0*pt.tanh(X/2.0)/pt.cosh(X/2.0), pt.exp((0.1+2.8j)*T))
f2 = pt.multiply(20-0.2*pt.pow(X, 2), pt.exp((-0.05+2.3j)*T))
f3 = pt.multiply(X, pt.exp(0.6j*T))
f4 = pt.multiply(20-0.2*pt.pow(X, 2), pt.exp((-0.05+2.3j)*T))

# Generate data based on superposition of the three functions
data = f1 + f2 + f3 + f4

# Add Gaussian noise to the data
noise_levels = [0.0, 0.5, 1.0, 2.0]
ranks = range(1, 6)

# Create a figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Plot compression rate vs. relative error for different noise levels (first subplot)
for noise_level in noise_levels:
    data_noisy = data + pt.normal(noise_level*pt.ones_like(X), noise_level*pt.ones_like(X))
    compression_rates = []
    relative_errors = []

    for rank in ranks:
        dmd = DynamicModeDecomposition(data_noisy, rank)
        compression_rates.append(dmd.compression_rate())
        relative_errors.append(np.mean(dmd.relative_error()))

    ax1.plot(compression_rates, relative_errors, marker='o', label=f'Noise Level: {noise_level}')

ax1.set_xlabel('Compression Rate')
ax1.set_ylabel('Relative Error')
ax1.set_title('Compression Rate vs. Relative Error')
ax1.legend()
ax1.grid(True)

# Plot rank vs. relative error for different noise levels (second subplot)
for noise_level in noise_levels:
    data_noisy = data + pt.normal(noise_level*pt.ones_like(X), noise_level*pt.ones_like(X))
    relative_errors = []

    for rank in ranks:
        dmd = DynamicModeDecomposition(data_noisy, rank)
        relative_errors.append(np.mean(dmd.relative_error()))

    ax2.plot(ranks, relative_errors, marker='o', label=f'Noise Level: {noise_level}')

ax2.set_xlabel('Rank')
ax2.set_ylabel('Relative Error')
ax2.set_title('Rank vs. Relative Error')
ax2.legend()
ax2.grid(True)

# Plot rank vs. reconstruction quality (relative error) for different noise levels (third subplot)
for noise_level in noise_levels:
    data_noisy = data + pt.normal(noise_level*pt.ones_like(X), noise_level*pt.ones_like(X))
    reconstruction_errors = []

    for rank in ranks:
        dmd = DynamicModeDecomposition(data_noisy, rank)
        data_DMD = dmd.reconstruct_data(t)
        quality = reconstruction_quality(data_noisy, data_DMD)
        reconstruction_errors.append(quality["mse"])

    ax3.plot(ranks, reconstruction_errors, marker='o', label=f'Noise Level: {noise_level}')

ax3.set_xlabel('Rank')
ax3.set_ylabel('Reconstruction Error')
ax3.set_title('Rank vs. Reconstruction Error')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()