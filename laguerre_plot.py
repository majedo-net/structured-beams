import matplotlib.pyplot as plt
import numpy as np
from math import factorial
from scipy.special import genlaguerre

# Set up beam paramater values

w_0 = 1.0  # Beam waist radius 
lda = 1    # Wavelength (lambda)
z_0 = 5.0  # Beam waist position
l = 5     # Azimuthal mode index
p = 5     # Radial mode index
propigation_distance = 10.0

freq = 1/lda # Frequency
r_length = (np.pi*(w_0**2))/lda # Rayleigh length(helps determine how fast the beam diverges)

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

z_slice = 5 #for now, I have put the xy slice at the beam waist #### THIS IS WHAT TO CHANGE TO GET THE XY AT A DIFFERENT Z SLICE ALONG THE BEAM
w_slice = w_0*np.sqrt(1+((z_slice-z_0)/r_length)**2) 
x_width=w_slice*(np.sqrt(2*p+np.abs(l)+1))      
y_width=w_slice*(np.sqrt(2*p+np.abs(l)+1)) 
x_set=np.linspace(-x_width,x_width,resolution)
y_set=np.linspace(-y_width,y_width,resolution)
arrayX,arrayY=np.meshgrid(x_set,y_set)
r_sq=arrayX**2 + arrayY**2
r=np.sqrt(r_sq)

# Calculate how beam evolves at each point along z-axis
# Propigation Direction version:
#zRel=bigZ-z_0 # makes sure that z coords are offset by beam waist
# w_z = w_0*np.sqrt( 1 + (zRel/r_length)**2) # calculates beam radius
# z_safe = np.where(zRel == 0, np.inf, zRel) # Uses np.where to prevent divide by zero errors in the next line
# R_z = z_safe * (1 + (r_length / z_safe)**2) # calculates wavefront radius of curvature
# psi_z = np.arctan(zRel/r_length)  # calculates Gouy phase


#Evaluate E-Field equation across coordinate grid
# E_xo = 1 #complex beam scaling typically = 1
k = (2*np.pi)/lda

# slice at distance z version:
r_z_slice= np.inf if z_slice==0 else z_slice* (1+(r_length/z_slice)**2)
phi=np.atan2(arrayY,arrayX) # computes azimuthal angle
psiz = (np.abs(l) + 2*p + 1)*np.atan2(z_slice,r_length) # computes mode-dependent Gouy Phase Shift

term1 = (np.sqrt(2*factorial(np.abs(p))/(np.pi*factorial(np.abs(p+np.abs(l)))))/w_slice)
term2 = ((r*np.sqrt(2)/w_slice)**np.abs(l))*np.exp(-r**2 / w_slice**2)*genlaguerre(p,np.abs(l))(2*r**2/w_slice**2)
term3 = np.exp((-1j*k*r**2) / (2*r_z_slice))
term4 = np.exp(-1j*l*phi)
term5 = np.exp(1j * psiz) # this was not in the example code given by matt, but adding it will mean that there is not a constant phase offset across the 2d slice
e_field_distribution = term1*term2*term3*term4*term5
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