# Gaussian Beam Intensity Plots
# Description: In this file, meep is used to propigate Gaussian beams through space, 
#              then, the intensity of the beams is plotted along the z axis 
#              (propigation direction) and x axis (transverse direction).
# Most Recent Update: 8/17/2026 (Daniel Schkade)


import meep as mp
import matplotlib.pyplot as plt
import numpy as np
import os

# Preparing place to hold output data 
#os.makedirs('gauss_results',exist_ok=True)

# Setting up variables for Gaussian beam generation
# The three variables below can be adjusted to change the intensity plots but other
# variables should not be carelessly changed
w_0 = 1.0     # Beam waist radius 
lda = 1.0      # Wavelength (lambda)
z_0 = 5.0      # Beam waist position
propigation_distance = 10.0

n = 1        # Material index of refraction (assuming vacuum for now so it will just be 1)
freq = 1/lda # Frequency

r_length = (np.pi*(w_0**2)*n)/lda # Rayleigh length
# The length of the propigation axis should be multiple Rayleigh lengths in order to get most major diffraction behavior within the cell

z_max=max(abs(z_0+(propigation_distance*2)),abs(z_0))
w_max = w_0*np.sqrt(1+(z_max/r_length)**2)  # Max beam waist (need to know so x and y are large enough)


# Set up computational cell
pml_thickness = 2.0 # possibly adjust later so this is calculated with changing lda
transverse_margin = 1.5*w_max  #may need to adjust to 2x w_max but for now this will make it faster
l_x = 2*transverse_margin + 2*pml_thickness     # X axis length
l_y = l_x                                     # Y axis length
l_z = 2*propigation_distance + 2*pml_thickness  # Z axis length

cell = mp.Vector3(l_x,l_y,l_z)


# Add PML boundaries to limit reflected waves (the code below is essentially the same as the pml setup for ms3d_opt.py)
pml_layers = [
    mp.PML(thickness=pml_thickness,direction=mp.Z,side=mp.ALL), # if error show up in code, consider where the source is in relation to the PML. It could cause an error if the source is to close to the pml right behind it in the cell
    mp.PML(thickness=pml_thickness,direction=mp.X,side=mp.ALL),
    mp.PML(thickness=pml_thickness,direction=mp.Y,side=mp.ALL)
    ]


# Define Gaussian beam source
resolution = 8 # consider adding code to calculate the best resolution if more efficncy or more precision are needed. 
#For now, 20 is used as it is a farily typical resultion on many meep examples
# Continous source will be used here because gaussian source uses a band of frequencies. 
# This may be implemented later but for now, continous soure will be better for analysis of a single frequency gaussian beam

source_position_z = 0.0
sources = [
    mp.GaussianBeam3DSource(mp.GaussianSource(frequency=freq,fwidth=0.2*freq),                        # could replace continous source with a gaussian source here I think
                             center=mp.Vector3(0,0,source_position_z),                   # where the source of the beam is located/centered
                             size=mp.Vector3(l_x-2*pml_thickness,l_y-2*pml_thickness,0), # Size of source, in this case, it is an xy plane filling the cell but not the PML
                             beam_x0=mp.Vector3(0,0,z_0-source_position_z),              # Distance between source and beam waist
                             beam_kdir=mp.Vector3(0,0,1),                                # Propigation direction set to positive z
                             beam_w0=w_0,
                             beam_E0=mp.Vector3(1,0,0)                                   # Must be parallel to source plane for transverse mode
                             )
    ]


# Set up simulation
# Currently has no symmetries but I need to go back and check if they are needed for this code (don't think so but haven't thought about it much)
sim = mp.Simulation(cell_size=cell,     
                    boundary_layers=pml_layers,
                    geometry=[],                 # Geometry left empty because I am currently using a vaccum and no simulated structure, if this changes, I will likley need to add a material grid because it would be a material with geometry
                    sources=sources,
                    resolution=resolution,
                    split_chunks_evenly=False    # Looking at the MEEP UI for python this is here just because it may make processing faster, can turn off if it messes stuff up
                    )

# Set up data collection (field output regions)
# First need data collection plane
collection_plane=mp.Volume(
                        center=mp.Vector3(0,0,2.5), # may need to chance this but for now it is fine, remember, this will be an xz plane so it is parallel, not perpendicular to the wave propigation
                        size=mp.Vector3(l_x-2*pml_thickness,0,(propigation_distance-2.5)) # 20 is propigation distance, possible adjustment to account for pml and source
                        )
# dft monitor addition
dft_monitor=sim.add_dft_fields( [mp.Ex,mp.Ey], freq, freq, 1, where=collection_plane)


# Code to run simulation
z_test_pos = (l_z/2)-pml_thickness-0.5*lda # by placing the test position for the stop_when_fields decayed right before the end of the region, it helps stop it at the right tome without being too early
sim.run(until_after_sources=mp.stop_when_fields_decayed(20,mp.Ex,mp.Vector3(0,0,z_test_pos),1e-5)) 

# Collect and Plot Results
Ex_out = sim.get_dft_array(dft_monitor,mp.Ex,0)
Ey_out = sim.get_dft_array(dft_monitor,mp.Ey,0)

intensity = (np.abs(Ex_out)**2) + (np.abs(Ey_out)**2) 

if intensity.ndim == 3:
    intensity = np.squeeze(intensity)      # this checks if the intensity array is 3D, which it should be. If it is, essentially makes the y slice very thin because only x and z directions need to be plotted in this project

collection_x_min = collection_plane.center.x - collection_plane.size.x/2
collection_x_max = collection_plane.center.x + collection_plane.size.x/2
collection_z_min = collection_plane.center.z - collection_plane.size.z/2
collection_z_max = collection_plane.center.z + collection_plane.size.z/2
x_coords = np.linspace(collection_x_min, collection_x_max, intensity.shape[0])
z_coords = np.linspace(collection_z_min, collection_z_max, intensity.shape[1])

plt.figure(figsize=(10,8))

plt.imshow(
    intensity.T,
    extent = [x_coords[0],x_coords[-1],z_coords[0],z_coords[-1]],
    origin="lower",
    cmap="inferno" 
)

plt.colorbar(label="Intensity")
plt.xlabel("Transverse Direction (X)")
plt.ylabel("Propigation Direction (Z)")   # the y axis of this plot is actually the z axis of the 3d area where the wave propigated
plt.title("Intensity of Gaussian Beam Over Distance")
plt.show()