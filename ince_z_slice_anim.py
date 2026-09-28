import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# --- Beam Parameter Setup ---
w_0 = 1.0  # Beam waist radius
lda = 1.0  # Wavelength
z_0 = 5.0  # Waist position
p = 6  # Order (degree) of beam
m = 2  # Azimuthal mode index
parity = 1  # 0 for even (C_p^m), 1 for odd (S_p^m)
e_param = 1.0  # Fixed eccentricity parameter
E_xo = 1.0
k = (2 * np.pi) / lda

r_length = (np.pi * (w_0**2)) / lda  # Rayleigh length
resolution = 200  # Grid points per axis
num_frames = 120
z_slices = np.linspace(0.0001, 9.999, num_frames)

# Calculate grid bounds based on maximum beam width across all z-slices
z_max_rel = np.max(np.abs(z_slices - z_0))
w_max = w_0 * np.sqrt(1 + (z_max_rel / r_length) ** 2)
x_width = w_max * np.sqrt(2 * p + 1)
y_width = w_max * np.sqrt(2 * p + 1)

x_set = np.linspace(-x_width, x_width, resolution)
y_set = np.linspace(-y_width, y_width, resolution)
arrayX, arrayY = np.meshgrid(x_set, y_set)
r_sq = arrayX**2 + arrayY**2

# Fixed focal parameter base
f_0 = w_0 * np.sqrt(e_param / 2.0)


def ince_poly(p, m, e_param, xi, eta, parity=0):
    """Calculates Ince polynomials using pure matrix eigensolvers."""
    q = e_param / 2.0
    is_even_mode = p % 2 == 0

    if is_even_mode:
        if parity == 0:  # Even C_p^m
            N = p // 2 + 1
            M = np.zeros((N, N), dtype=float)
            for r_idx in range(N):
                M[r_idx, r_idx] = (2 * r_idx) ** 2
                if r_idx < N - 1:
                    factor = q * (p + 2 * r_idx + 2)
                    if r_idx == 0:
                        factor *= 2
                    M[r_idx, r_idx + 1] = factor
                if r_idx > 0:
                    M[r_idx, r_idx - 1] = q * (p - 2 * r_idx + 2)

            vals, vec = np.linalg.eig(M)
            vals, vec = np.real(vals), np.real(vec)
            idx = np.argsort(vals)
            A = vec[:, idx][:, m // 2]
            A /= np.linalg.norm(A)

            eta_term = sum(A[r_i] * np.cos(2 * r_i * eta) for r_i in range(N))
            xi_term = sum(A[r_i] * np.cosh(2 * r_i * xi) for r_i in range(N))

        else:  # Odd S_p^m
            N = p // 2
            M = np.zeros((N, N), dtype=float)
            for k_idx in range(N):
                r_val = k_idx + 1
                M[k_idx, k_idx] = (2 * r_val) ** 2
                if k_idx < N - 1:
                    M[k_idx, k_idx + 1] = q * (p + 2 * r_val + 2)
                if k_idx > 0:
                    M[k_idx, k_idx - 1] = q * (p - 2 * r_val + 2)

            vals, vec = np.linalg.eig(M)
            vals, vec = np.real(vals), np.real(vec)
            idx = np.argsort(vals)
            B = vec[:, idx][:, (m // 2) - 1]
            B /= np.linalg.norm(B)

            eta_term = sum(
                B[k_i] * np.sin(2 * (k_i + 1) * eta) for k_i in range(N)
            )
            xi_term = sum(
                B[k_i] * np.sinh(2 * (k_i + 1) * xi) for k_i in range(N)
            )

    else:  # Odd-Odd modes
        N = (p + 1) // 2
        M = np.zeros((N, N), dtype=float)

        if parity == 0:  # Odd C_p^m
            for r_idx in range(N):
                M[r_idx, r_idx] = (2 * r_idx + 1) ** 2 + (q if r_idx == 0 else 0)
                if r_idx < N - 1:
                    M[r_idx, r_idx + 1] = q * (p + 2 * r_idx + 3)
                if r_idx > 0:
                    M[r_idx, r_idx - 1] = q * (p - 2 * r_idx + 1)

            vals, vec = np.linalg.eig(M)
            vals, vec = np.real(vals), np.real(vec)
            idx = np.argsort(vals)
            A = vec[:, idx][:, (m - 1) // 2]
            A /= np.linalg.norm(A)

            eta_term = sum(
                A[r_i] * np.cos((2 * r_i + 1) * eta) for r_i in range(N)
            )
            xi_term = sum(
                A[r_i] * np.cosh((2 * r_i + 1) * xi) for r_i in range(N)
            )

        else:  # Odd S_p^m
            for r_idx in range(N):
                M[r_idx, r_idx] = (2 * r_idx + 1) ** 2 - (q if r_idx == 0 else 0)
                if r_idx < N - 1:
                    M[r_idx, r_idx + 1] = q * (p + 2 * r_idx + 3)
                if r_idx > 0:
                    M[r_idx, r_idx - 1] = q * (p - 2 * r_idx + 1)

            vals, vec = np.linalg.eig(M)
            vals, vec = np.real(vals), np.real(vec)
            idx = np.argsort(vals)
            B = vec[:, idx][:, (m - 1) // 2]
            B /= np.linalg.norm(B)

            eta_term = sum(
                B[k_i] * np.sin((2 * k_i + 1) * eta) for k_i in range(N)
            )
            xi_term = sum(
                B[k_i] * np.sinh((2 * k_i + 1) * xi) for k_i in range(N)
            )

    return eta_term * xi_term


# --- Initial Figure Setup ---
blank_grid = np.zeros_like(arrayX)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

im1 = ax1.imshow(
    blank_grid,
    extent=[arrayX.min(), arrayX.max(), arrayY.min(), arrayY.max()],
    origin="lower",
    cmap="inferno",
    vmin=0,
    vmax=1,
)
title = ax1.set_title("")
ax1.set_xlabel("x position")
ax1.set_ylabel("y position")
fig.colorbar(im1, ax=ax1, label="Normalized Intensity")

im2 = ax2.imshow(
    blank_grid,
    extent=[arrayX.min(), arrayX.max(), arrayY.min(), arrayY.max()],
    origin="lower",
    cmap="twilight",
    vmin=-np.pi,
    vmax=np.pi,
)
ax2.set_title("Beam Phase")
ax2.set_xlabel("x position")
ax2.set_ylabel("y position")
fig.colorbar(im2, ax=ax2, label="Phase (rad)")

plt.tight_layout()


# --- Frame Update Function ---
def update(frame):
    z_slice = z_slices[frame]
    z_rel = z_slice - z_0

    # Dynamic z-dependent beam parameters
    w_slice = w_0 * np.sqrt(1 + (z_rel / r_length) ** 2)
    r_z_slice = np.inf if z_rel == 0 else z_rel * (1 + (r_length / z_rel) ** 2)
    psiz = -(p + 1) * np.arctan2(z_rel, r_length)

    # Dynamic focal distance & elliptical coordinate scaling
    f_z = f_0 * (w_slice / w_0)
    z_scaled = (arrayX + 1j * arrayY) / f_z
    cosh_transform = np.arccosh(z_scaled)
    xi = np.abs(np.real(cosh_transform))
    eta = np.imag(cosh_transform) % (2 * np.pi)

    # Dynamic field components
    term1 = E_xo * w_0 / w_slice
    gaussian_envelope = np.exp(-r_sq / (w_slice**2))
    phase_curvature = 0 if z_rel == 0 else (k * r_sq) / (2 * r_z_slice)
    phase_term = np.exp(1j * (-k * z_slice + psiz + phase_curvature))

    # Ince polynomial and total electric field
    spatial_envelope = ince_poly(p, m, e_param, xi, eta, parity=parity)
    e_field = term1 * gaussian_envelope * spatial_envelope * phase_term

    intensity = np.abs(e_field) ** 2
    if np.max(intensity) > 0:
        intensity /= np.max(intensity)

    phase = np.angle(e_field)

    # In-place graphics update
    im1.set_array(intensity)
    im2.set_array(phase)
    p_label = "e" if parity == 0 else "o"
    title.set_text(
        f"Ince Beam Intensity ($IG_{{{p},{m}}}^{{{p_label}}}$, z = {z_slice:.2f})"
    )

    return im1, im2, title


# --- Run Animation ---
anim = FuncAnimation(
    fig, update, frames=num_frames, interval=40, blit=False, repeat=True
)

plt.show()


# To save as an MP4 or GIF, uncomment one of these lines:
# anim.save("ince_z_transition.mp4", writer="ffmpeg", fps=25)
# anim.save("ince_z_transition.gif", writer="pillow", fps=25)