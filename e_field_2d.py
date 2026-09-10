import matplotlib.pyplot as plt
import numpy as np

# Set up beam paramater values

w_0 = 1.0     # Beam waist radius 
lda = 1      # Wavelength (lambda)
z_0 = 5.0  # Beam waist position
propigation_distance = 10.0 #may or may not need. If doing e-field in the same way as gaussbeam, then yes, if taking a cross section at one z value, possibly not

freq = 1/lda # Frequency
r_length = (np.pi*(w_0**2))/lda # Rayleigh length(helps determine how fast the beam diverges)


# Establish Viewing Plane

# if doing same way as gaussbeam, need a transverse axis (x) and propigation axis (z)
# the two lines below are just taken from gaussbeam.py because this part of setup is the same
z_max=max(abs(z_0+(propigation_distance*2)),abs(z_0))
z_min=0 #assuming that the beam source is at at z=0
w_max = w_0*np.sqrt(1+((z_max-propigation_distance)/r_length)**2) #max beam waist (need to know to make x and y wide enough)
x_span=1.005*w_max  #span of transerse window. Can make larger but for now this will be faster
resolution=200 # Number of points along each axis. May need much higher but for speed I will just do this
x_list=np.linspace(-x_span,x_span,resolution)
z_list=np.linspace(z_min,z_max,resolution)
bigX, bigZ =np.meshgrid(x_list,z_list) #turns the two lists into a 2D coordinate grid where bigX represents transverse and bigZ represents propigation distance at each point

# for a static cross section e-field do transverse axis (x) and transverse axis (y) at a specific propigation distance (z) (commented out for now as I finish the other version)

z_slice = 5 #for now, I have put the xy slice at the beam waist #### THIS IS WHAT TO CHANGE TO GET THE XY AT A DIFFERENT Z SLICE ALONG THE BEAM
w_slice = w_0*np.sqrt(1+((z_slice-z_0)/r_length)**2) # I haven't checked this math yet but I am infering that I can adjust the w_max equation by adjusting my z_max to only where z_slice is to save space
x_width=w_slice*1.2
y_width=w_slice*1.2
x_set=np.linspace(-x_width,x_width,resolution)
y_set=np.linspace(-y_width,y_width,resolution)
arrayX,arrayY=np.meshgrid(x_set,y_set)
r_sq=arrayX**2 + arrayY**2


# Calculate how beam evolves at each point along z-axis
# Propigation Direction version:
zRel=bigZ-z_0 # makes sure that z coords are offset by beam waist
w_z = w_0*np.sqrt( 1 + (zRel/r_length)**2) # calculates beam radius
z_safe = np.where(zRel == 0, np.inf, zRel) # Uses np.where to prevent divide by zero errors in the next line
R_z = z_safe * (1 + (r_length / z_safe)**2) # calculates wavefront radius of curvature
psi_z = np.arctan(zRel/r_length)  # calculates Gouy phase


#Evaluate E-Field equation across coordinate grid
E_xo = 1 #complex beam scaling typically = 1
k = (2*np.pi)/lda

E_gauss= E_xo * (w_0/w_z) * np.exp((-bigX**2)/w_z**2) * np.exp(1j*(-k*zRel+psi_z-(k*(bigX**2)/(2*R_z))))  # May need to check for accuracy

# slice at distance z version:
r_z_slice= z_slice* (1+(r_length/z_slice)**2)
i_0 =1 # complex beam scaling still, just for this slice at distance z version
intensity_slice=i_0 * (w_0/w_slice) ** 2 * np.exp(-2 * r_sq/w_slice**2)

#Calculate Intensity across grid
intensity_stats=np.abs(E_gauss)**2

#Pass coords and calculated field values to matplot and plot them
plt.figure(figsize=(10,4))

plt.pcolormesh(bigZ,bigX, intensity_stats,shading='auto',cmap='inferno') # consider multiplying z_list and x_list by 1e3 to change to mm so it is easier to read
plt.colorbar(label='Intensity ($|E|^2$)')
plt.xlabel('Propigation Distance ($z$)')
plt.ylabel('Transverse Position ($x$)')
plt.title("Gaussian Beam 2d Propigation")
plt.tight_layout()
plt.show()


# Plotting the slice z version
plt.figure(figsize=(7,6))
im= plt.imshow(
    intensity_slice,
    extent=[arrayX.min(),arrayX.max(),arrayY.min(),arrayY.max()],
    origin="lower",
    cmap="inferno"
)

plt.title("Intensity at Distance 'z' Along Beam Propigation Path")
plt.colorbar(im,label="Intensity")
plt.xlabel("x position")
plt.ylabel("y position")
plt.tight_layout()
plt.show()