import matplotlib.pyplot as plt
import numpy as np

# Set up beam paramater values

w_0 = 1.0  # Beam waist radius 
lda = 1    # Wavelength (lambda)
z_0 = 5.0  # Beam waist position
m = 5      # azimuthal mode index  
p = 5      # represents order (degree) of beam (must be even when m is even and odd when m is odd)
e_param= 2 # ellipticity paramater
parity = 0 # 0 for even, 1 for odd   
propigation_distance = 10.0

freq = 1/lda                    # Frequency
r_length = (np.pi*(w_0**2))/lda # Rayleigh length(helps determine how fast the beam diverges)
f_0 = w_0*np.sqrt(e_param/2)    # Focal parameter

# Establish Viewing Plane
resolution=200 # Number of points along each axis

# if doing same way as gaussbeam, need a transverse axis (x) and propigation axis (z)
# z_max=max(abs(z_0+(propigation_distance*2)),abs(z_0))
# z_min=0 #assuming that the beam source is at at z=0
# w_max = w_0*np.sqrt(1+((z_max-propigation_distance)/r_length)**2)
# x_span=1.005*w_max  #span of transerse window. Can make larger but for now this will be faster
# x_list=np.linspace(-x_span,x_span,resolution)
# z_list=np.linspace(z_min,z_max,resolution)
# bigX, bigZ =np.meshgrid(x_list,z_list) #turns the two lists into a 2D coordinate grid where bigX represents transverse and bigZ represents propigation distance at each point

# for a static cross section e-field do transverse axis (x) and transverse axis (y) at a specific propigation distance (z)

z_slice = 7 #for now, I have put the xy slice at the beam waist #### THIS IS WHAT TO CHANGE TO GET THE XY AT A DIFFERENT Z SLICE ALONG THE BEAM
z_rel=z_slice-z_0
w_slice = w_0*np.sqrt(1+((z_slice-z_0)/r_length)**2) 
f_z = f_0*w_slice/w_0 # focal parameter at slice
x_width=w_slice*(np.sqrt(2*p+1)) + f_z      
y_width=w_slice*(np.sqrt(2*p+1)) + f_z
x_set=np.linspace(-x_width,x_width,resolution)
y_set=np.linspace(-y_width,y_width,resolution)
arrayX,arrayY=np.meshgrid(x_set,y_set)
r_sq=arrayX**2 + arrayY**2
r=np.sqrt(r_sq)

# Process to transform to elliptical coords
z_scaled = (arrayX+1j*arrayY)/f_z
cosh_transform = np.arccosh(z_scaled)
xi = np.abs(np.real(cosh_transform))
eta = np.imag(cosh_transform) % (2 * np.pi)

# Calculate how beam evolves at each point along z-axis
# Propigation Direction version:
#zRel=bigZ-z_0 # makes sure that z coords are offset by beam waist
# w_z = w_0*np.sqrt( 1 + (zRel/r_length)**2) # calculates beam radius
# z_safe = np.where(zRel == 0, np.inf, zRel) # Uses np.where to prevent divide by zero errors in the next line
# R_z = z_safe * (1 + (r_length / z_safe)**2) # calculates wavefront radius of curvature
# psi_z = np.arctan(zRel/r_length)  # calculates Gouy phase


#Evaluate E-Field equation across coordinate grid
E_xo = 1 #complex beam scaling typically = 1
k = (2*np.pi)/lda

#disclamer, used gen-ai assistance to figure out how to make this ince polynomial function
def ince_poly(p, m, e_param, xi, eta, parity=0):
    """Calculates Ince polynomials for even-even or odd-odd modes.

    parity=0 -> Even (C_p^m), parity=1 -> Odd (S_p^m)
    """
    if (p - m) % 2 != 0:
        raise ValueError(
            "p and m must have the same parity (p - m must be even)."
        )

    q = e_param / 2.0
    is_even_mode = p % 2 == 0

    if is_even_mode:
        if parity == 0:  # Even-Even C_p^m
            N = p // 2 + 1
            M = np.zeros((N, N), dtype=float)
            for r in range(N):
                M[r, r] = (2 * r) ** 2
                if r < N - 1:
                    factor = q * (p + 2 * r + 2)
                    if r == 0:
                        factor *= 2
                    M[r, r + 1] = factor
                if r > 0:
                    M[r, r - 1] = q * (p - 2 * r + 2)  # Fixed index [r, r-1]

            _, vec = np.linalg.eig(M)
            A = vec[:, np.argsort(_)][:, m // 2]
            A /= np.linalg.norm(A)

            eta_term = sum(A[r] * np.cos(2 * r * eta) for r in range(N))
            xi_term = sum(A[r] * np.cosh(2 * r * xi) for r in range(N))

        else:  # Even-Even S_p^m (m >= 2)
            if m == 0:
                raise ValueError("Odd mode S_p^m does not exist for m = 0.")
            N = p // 2
            M = np.zeros((N, N), dtype=float)
            for k in range(N):
                r = k + 1
                M[k, k] = (2 * r) ** 2
                if k < N - 1:
                    M[k, k + 1] = q * (p + 2 * r + 2)
                if k > 0:
                    M[k, k - 1] = q * (p - 2 * r + 2)  # Fixed index [k, k-1]

            _, vec = np.linalg.eig(M)
            B = vec[:, np.argsort(_)][:, (m // 2) - 1]
            B /= np.linalg.norm(B)

            eta_term = sum(
                B[k] * np.sin(2 * (k + 1) * eta) for k in range(N)
            )
            xi_term = sum(
                B[k] * np.sinh(2 * (k + 1) * xi) for k in range(N)
            )

    else:
        # Odd-Odd modes (p odd, m odd)
        N = (p + 1) // 2
        M = np.zeros((N, N), dtype=float)

        if parity == 0:  # Odd-Odd C_p^m
            for r in range(N):
                M[r, r] = (2 * r + 1) ** 2 + (q if r == 0 else 0)
                if r < N - 1:
                    M[r, r + 1] = q * (p + 2 * r + 3)
                if r > 0:
                    M[r, r - 1] = q * (p - 2 * r + 1)  # Fixed index [r, r-1]

            _, vec = np.linalg.eig(M)
            A = vec[:, np.argsort(_)][:, (m - 1) // 2]
            A /= np.linalg.norm(A)

            eta_term = sum(
                A[r] * np.cos((2 * r + 1) * eta) for r in range(N)
            )
            xi_term = sum(
                A[r] * np.cosh((2 * r + 1) * xi) for r in range(N)
            )

        else:  # Odd-Odd S_p^m
            for r in range(N):
                M[r, r] = (2 * r + 1) ** 2 - (q if r == 0 else 0)
                if r < N - 1:
                    M[r, r + 1] = q * (p + 2 * r + 3)
                if r > 0:
                    M[r, r - 1] = q * (p - 2 * r + 1)  # Fixed index [r, r-1]

            _, vec = np.linalg.eig(M)
            B = vec[:, np.argsort(_)][:, (m - 1) // 2]
            B /= np.linalg.norm(B)

            eta_term = sum(
                B[r] * np.sin((2 * r + 1) * eta) for r in range(N)
            )
            xi_term = sum(
                B[r] * np.sinh((2 * r + 1) * xi) for r in range(N)
            )

    return eta_term, xi_term
    """Calculates Ince polynomials for even-even or odd-odd modes.

    parity=0 -> Even (C_p^m), parity=1 -> Odd (S_p^m)
    """
    if (p - m) % 2 != 0:
        raise ValueError("p and m must have the same parity (p - m must be even).")

    q = e_param / 2.0
    is_even_mode = p % 2 == 0

    if is_even_mode:
        if parity == 0:  # Even-Even C_p^m
            N = p // 2 + 1
            M = np.zeros((N, N), dtype=float)
            for r in range(N):
                M[r, r] = (2 * r) ** 2
                if r < N - 1:
                    factor = q * (p + 2 * r + 2)
                    if r == 0:
                        factor *= 2
                    M[r, r + 1] = factor
                if r > 0:
                    M[r - 1, r] = q * (p - 2 * r + 2)

            _, vec = np.linalg.eig(M)
            A = vec[:, np.argsort(_)][:, m // 2]
            A /= np.linalg.norm(A)

            eta_term = sum(A[r] * np.cos(2 * r * eta) for r in range(N))
            xi_term = sum(A[r] * np.cosh(2 * r * xi) for r in range(N))

        else:  # Even-Even S_p^m (m >= 2)
            if m == 0:
                raise ValueError("Odd mode S_p^m does not exist for m = 0.")
            N = p // 2
            M = np.zeros((N, N), dtype=float)
            for k in range(N):
                r = k + 1  # Harmonic index
                M[k, k] = (2 * r) ** 2
                if k < N - 1:
                    M[k, k + 1] = q * (p + 2 * r + 2)
                if k > 0:
                    M[k - 1, k] = q * (p - 2 * r + 2)

            _, vec = np.linalg.eig(M)
            B = vec[:, np.argsort(_)][:, (m // 2) - 1]
            B /= np.linalg.norm(B)

            eta_term = sum(
                B[k] * np.sin(2 * (k + 1) * eta) for k in range(N)
            )
            xi_term = sum(
                B[k] * np.sinh(2 * (k + 1) * xi) for k in range(N)
            )

    else:
        # Odd-Odd modes (p odd, m odd)
        N = (p + 1) // 2
        M = np.zeros((N, N), dtype=float)

        if parity == 0:  # Odd-Odd C_p^m
            for r in range(N):
                M[r, r] = (2 * r + 1) ** 2 + (q if r == 0 else 0)
                if r < N - 1:
                    M[r, r + 1] = q * (p + 2 * r + 3)
                if r > 0:
                    M[r - 1, r] = q * (p - 2 * r + 1)

            _, vec = np.linalg.eig(M)
            A = vec[:, np.argsort(_)][:, (m - 1) // 2]
            A /= np.linalg.norm(A)

            eta_term = sum(
                A[r] * np.cos((2 * r + 1) * eta) for r in range(N)
            )
            xi_term = sum(
                A[r] * np.cosh((2 * r + 1) * xi) for r in range(N)
            )

        else:  # Odd-Odd S_p^m
            for r in range(N):
                M[r, r] = (2 * r + 1) ** 2 - (q if r == 0 else 0)
                if r < N - 1:
                    M[r, r + 1] = q * (p + 2 * r + 3)
                if r > 0:
                    M[r - 1, r] = q * (p - 2 * r + 1)

            _, vec = np.linalg.eig(M)
            B = vec[:, np.argsort(_)][:, (m - 1) // 2]
            B /= np.linalg.norm(B)

            eta_term = sum(
                B[r] * np.sin((2 * r + 1) * eta) for r in range(N)
            )
            xi_term = sum(
                B[r] * np.sinh((2 * r + 1) * xi) for r in range(N)
            )

    return eta_term, xi_term

# slice at distance z version:
r_z_slice= np.inf if z_rel==0 else z_rel* (1+(r_length/z_rel)**2)
psiz = -(p+1)*np.atan2(z_rel,r_length) # computes mode-dependent Gouy Phase Shift
term1 = E_xo*w_0/w_slice
C_eta, C_xi = ince_poly(p, m, e_param, xi, eta, parity=parity)
ince_envelope = C_eta * C_xi
term4 = np.exp(-r_sq / (w_slice**2))
phase_curvature = 0 if z_rel == 0 else (k * r_sq) / (2 * r_z_slice)
term5 = np.exp(1j*(-k*z_slice+psiz+phase_curvature))

e_field_distribution = term1*ince_envelope*term4*term5
intensity_field=np.abs(e_field_distribution)**2
phase_field=np.angle(e_field_distribution)

#Pass coords and calculated field values to matplot and plot them
# plt.figure(figsize=(10,4))

# plt.pcolormesh(bigZ,bigX, intensity_stats,shading='auto',cmap='inferno') # consider multiplying z_list and x_list by 1e3 to change to mm so it is easier to read
# plt.colorbar(label='Intensity ($|E|^2$)')
# plt.xlabel('Propigation Distance ($z$)')
# plt.ylabel('Transverse Position ($x$)')
# plt.title("Gaussian Beam 2d Propigation")
# plt.tight_layout()
# plt.show()


# Plotting the slice z version
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# Plot Intensity
im1 = ax1.imshow(
    intensity_field,
    extent=[arrayX.min(), arrayX.max(), arrayY.min(), arrayY.max()],
    origin="lower",
    cmap="inferno",
)
ax1.set_title("Beam Intensity")
ax1.set_xlabel("x position")
ax1.set_ylabel("y position")
fig.colorbar(im1, ax=ax1, label="Intensity ($|E|^2$)")

# Plot Phase
im2 = ax2.imshow(
    phase_field,
    extent=[arrayX.min(), arrayX.max(), arrayY.min(), arrayY.max()],
    origin="lower",
    cmap="twilight",
)
ax2.set_title("Beam Phase")
ax2.set_xlabel("x position")
ax2.set_ylabel("y position")
fig.colorbar(im2, ax=ax2, label="Phase (rad)")

plt.tight_layout()
plt.show()