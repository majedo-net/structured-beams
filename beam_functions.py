import numpy as np
from math import factorial
from scipy.special import genlaguerre
from scipy.special import hermite

def gauss_xy(w_0,lda,z_0,z_slice,i_0=1,resolution=200):
    r_length = (np.pi*(w_0**2))/lda # Rayleigh length(helps determine how fast the beam diverges)
    w_slice = w_0*np.sqrt(1+((z_slice-z_0)/r_length)**2)
    x_width=w_slice*1.2
    y_width=w_slice*1.2
    x_set=np.linspace(-x_width,x_width,resolution)
    y_set=np.linspace(-y_width,y_width,resolution)
    arrayX,arrayY=np.meshgrid(x_set,y_set)
    r_sq=arrayX**2 + arrayY**2
    # k = (2*np.pi)/lda
    # r_z_slice = z_slice* (1+(r_length/z_slice)**2)
    intensity_slice = i_0 * (w_0/w_slice) ** 2 * np.exp(-2 * r_sq/w_slice**2)
    return intensity_slice,arrayX,arrayY

def gauss_xz(w_0,lda,z_0,prop_d,z_min=0,E_xo=1,resolution=200):
    r_length = (np.pi*(w_0**2))/lda # Rayleigh length(helps determine how fast the beam diverges)
    z_max=max(abs(z_0+(prop_d*2)),abs(z_0))
    w_max = w_0*np.sqrt(1+((z_max-prop_d)/r_length)**2) 
    x_span=1.005*w_max
    resolution=200 
    x_list=np.linspace(-x_span,x_span,resolution)
    z_list=np.linspace(z_min,z_max,resolution)
    bigX, bigZ =np.meshgrid(x_list,z_list)
    zRel=bigZ-z_0
    w_z = w_0*np.sqrt( 1 + (zRel/r_length)**2) 
    z_safe = np.where(zRel == 0, np.inf, zRel) 
    R_z = z_safe * (1 + (r_length / z_safe)**2) 
    psi_z = np.arctan(zRel/r_length)
    k = (2*np.pi)/lda
    E_gauss= E_xo * (w_0/w_z) * np.exp((-bigX**2)/w_z**2) * np.exp(1j*(-k*zRel+psi_z-(k*(bigX**2)/(2*R_z))))
    intensity_stats=np.abs(E_gauss)**2
    return intensity_stats,bigX,bigZ

def laguerre_xy(w_0,lda,z_0,z_slice,l=1,p=1,resolution=200):
    r_length = (np.pi*(w_0**2))/lda
    w_slice = w_0*np.sqrt(1+((z_slice-z_0)/r_length)**2) 
    x_width=w_slice*(np.sqrt(2*p+np.abs(l)+1))      
    y_width=w_slice*(np.sqrt(2*p+np.abs(l)+1)) 
    x_set=np.linspace(-x_width,x_width,resolution)
    y_set=np.linspace(-y_width,y_width,resolution)
    arrayX,arrayY=np.meshgrid(x_set,y_set)
    r_sq=arrayX**2 + arrayY**2
    r=np.sqrt(r_sq)
    k = (2*np.pi)/lda
    r_z_slice= np.inf if z_slice==0 else z_slice* (1+(r_length/z_slice)**2)
    phi=np.atan2(arrayY,arrayX)
    psiz = (np.abs(l) + 2*p + 1)*np.atan2(z_slice,r_length)
    term1 = (np.sqrt(2*factorial(np.abs(p))/(np.pi*factorial(np.abs(p+np.abs(l)))))/w_slice)
    term2 = ((r*np.sqrt(2)/w_slice)**np.abs(l))*np.exp(-r**2 / w_slice**2)*genlaguerre(p,np.abs(l))(2*r**2/w_slice**2)
    term3 = np.exp((-1j*k*r**2) / (2*r_z_slice))
    term4 = np.exp(-1j*l*phi)
    term5 = np.exp(1j * psiz)
    e_field_distribution = term1*term2*term3*term4*term5
    intensity_field=np.abs(e_field_distribution)**2
    phase_field=np.angle(e_field_distribution)
    return intensity_field,phase_field,arrayX,arrayY

def hermite_xy(w_0,lda,z_0,z_slice,E_xo=1,m=1,n=1,resolution=200):
    r_length = (np.pi*(w_0**2))/lda
    z_rel=z_slice-z_0
    w_slice = w_0*np.sqrt(1+((z_slice-z_0)/r_length)**2) 
    x_width=w_slice*(np.sqrt(2*n+1))      
    y_width=w_slice*(np.sqrt(2*m+1)) 
    x_set=np.linspace(-x_width,x_width,resolution)
    y_set=np.linspace(-y_width,y_width,resolution)
    arrayX,arrayY=np.meshgrid(x_set,y_set)
    r_sq=arrayX**2 + arrayY**2
    r=np.sqrt(r_sq)
    k = (2*np.pi)/lda
    r_z_slice= np.inf if z_rel==0 else z_rel* (1+(r_length/z_rel)**2)
    psiz = -(n+m+1)*np.atan2(z_rel,r_length)
    term1 = E_xo*w_0/w_slice
    term2= hermite(m)(np.sqrt(2)*arrayX/w_slice)
    term3=hermite(n)(np.sqrt(2)*arrayY/w_slice)
    term4 = np.exp(-r_sq / (w_slice**2))
    phase_curvature = 0 if np.isinf(r_z_slice) else (k * r_sq) / (2 * r_z_slice)
    term5 = np.exp(-1j * (k * z_slice + phase_curvature - psiz))
    e_field_distribution = term1*term2*term3*term4*term5
    intensity_field=np.abs(e_field_distribution)**2
    phase_field=np.angle(e_field_distribution)
    return intensity_field, phase_field,arrayX,arrayY

def ince_xy (w_0,lda,z_0,z_slice,E_xo=1,m=1,p=1,e_param=1,parity=0,resolution=200):
    r_length = (np.pi*(w_0**2))/lda # Rayleigh length(helps determine how fast the beam diverges)
    f_0 = w_0*np.sqrt(e_param/2)
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
    z_scaled = (arrayX+1j*arrayY)/f_z
    cosh_transform = np.arccosh(z_scaled)
    xi = np.abs(np.real(cosh_transform))
    eta = np.imag(cosh_transform) % (2 * np.pi)
    k = (2*np.pi)/lda
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
    return intensity_field,phase_field,arrayX,arrayY




    
