import meep as mp
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product
from autograd.tracer import getval
import meep.adjoint as mpa
import os
import nlopt
from autograd.scipy.special import jn
from pub_cmap import paper_cmap

plt.style.use(['ieee','no-latex'])

os.makedirs('results',exist_ok=True)

lda0 = 1.0 # cm = 30ghz
lambda_min = lda0 - 0.1 
lambda_max = lda0 + 0.1

freq0 = 1 / lda0
fmin = 1 / lambda_max
fmax = 1 / lambda_min
freqs = np.linspace(fmin, fmax, 101)

fwidth = (fmax - fmin)
run_time = 100 / fwidth
resolution = 20 

interval = 1/resolution

buffer_z = np.round(lda0/interval)*interval
length_z = 6*buffer_z

n_radix = 2.8**2 # Rogers Radix

DRX = 30
DRY = 0.0
height = 2
filter_radius = 0.05

NA = 0.5
FF_DIST = (DRX/2) / NA * np.sqrt(1 - NA**2)  # focal length
#FF_DIST = (2*DRX**2)/lda0
print(f'{FF_DIST=}')

DR_CM = mp.Vector3(DRX,height)

DR_RES = 10

DR_N_X = int(round(DR_CM.x * DR_RES)) + 1
DR_N_H = int(round(DR_CM.y * DR_RES)) + 1

cell = mp.Vector3(DRX + 2*buffer_z, height+length_z)

pml_layers = [
    #mp.PML(thickness=buffer_z,direction=mp.Z,side=mp.ALL),
    mp.PML(thickness=buffer_z,direction=mp.X,side=mp.ALL),
    mp.PML(thickness=buffer_z,direction=mp.Y,side=mp.ALL)
    ]

src1 = mp.GaussianSource(frequency=freq0,fwidth=fwidth,is_integrated=True)
sources = [mp.Source(src1,
                    component=mp.Ez,
                    size=mp.Vector3(DRX,0),
                    center=mp.Vector3(0,-(height/2) -(buffer_z)))]

mat_grid = mp.MaterialGrid(
    grid_size= mp.Vector3(DR_N_X,DR_N_H),
    medium1=mp.air,
    medium2=mp.Medium(index=n_radix),
    #beta=4,
    #eta=0.5,
    do_averaging=True,
    #weights=np.load('meep/rfms/results/fmin.npy'),
    weights=0.5*np.ones((DR_N_X,DR_N_H)),
    grid_type="U_MEAN"
)
mat_grid_region = mpa.DesignRegion(
    mat_grid,
    volume=mp.Volume(
        center=mp.Vector3(0,0),
        size=DR_CM
    )
)
atoms = mp.Block(
    material=mat_grid,
    center=mat_grid_region.center,
    size=mat_grid_region.size
)

geometry = [atoms]
symmetries = [mp.Mirror(direction=mp.X)]

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    symmetries=symmetries,
                    split_chunks_evenly=False,
                    resolution=resolution)

RUN_START = True
if RUN_START:

    n2f_obj = sim.add_near2far(
        freq0, 0, 1,
        mp.Near2FarRegion(
            center=mp.Vector3(0,height/2 + buffer_z), 
            size=mp.Vector3(DRX,0), weight=+1
            )
        )

    if mp.am_really_master():
        fig,ax1=plt.subplots(1,1,figsize=(15,15))
        sim.plot2D(ax=ax1,output_plane=mp.Volume(center=mp.Vector3(),size=mp.Vector3(cell.x,cell.y)))
        fig.tight_layout()
        fig.savefig(f'results/meta_geo_start.png')

    sim.run(until_after_sources=mp.stop_when_energy_decayed(1000, 1e-5))
    ffields = sim.get_farfields(n2f_obj, resolution=20, center=mp.Vector3(0,FF_DIST), size=mp.Vector3(10*lda0))
    ifar = np.abs(ffields['Ez'])**2

    xcfields = sim.get_farfields(n2f_obj, resolution=20, center=mp.Vector3(0,FF_DIST), size=mp.Vector3(10*lda0,0.5*FF_DIST))
    ixc = np.abs(xcfields['Ez'])**2
    if mp.am_really_master():
        fig, ax = plt.subplots(1,1,figsize=(5,5)) 
        ax.plot(ifar)
        fig.tight_layout()
        fig.savefig('results/metalens_farfield_start.png')

        fig, ax = plt.subplots(1,1,figsize=(5,5)) 
        ax.pcolormesh(ixc.T, cmap=paper_cmap, shading='gouraud')
        fig.tight_layout()
        fig.savefig('results/metalens_farfield_xc_start.png')

def mapping(x, eta, beta):
    # filter
    filtered_field = mpa.conic_filter(
        x,
        filter_radius,
        DRX,
        height,
        DR_RES,
    )

    # projection
    projected_field = mpa.tanh_projection(filtered_field, beta, eta)

    projected_field = (
        npa.flipud(projected_field) + projected_field
    ) / 2  # left-right symmetry

    # interpolate to actual materials
    return projected_field.flatten()

far_x =npa.linspace(-5*lda0, 5*lda0, 301)
far_pts = [mp.Vector3(x,FF_DIST) for x in far_x]

NearRegions = [
    mp.Near2FarRegion(
        center = mp.Vector3(0,height/2 + buffer_z),
        size=mp.Vector3(DRX,0), weight=+1
    )
]

FarFields = mpa.Near2FarFields(sim, NearRegions, far_pts)

obj_list = [FarFields]

iteration = 0
fmin = 1e10
eta_i = 0.5

def J1(FF):
    global iteration
    obj_fields = jn(0,2*npa.pi*npa.abs(far_x))
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    ax.plot(np.real(getval(obj_fields)),label='objective')
    ef = FF[:,0,2]
    ef = ef / npa.max(npa.abs(ef))
    cost = npa.linalg.norm(ef - obj_fields)**2

    ax.plot(np.real(getval(ef)),label='simmed')
    ax.legend()
    ax.set_title(f'Cost = {getval(cost):3.3f}')
    fig.suptitle(f'beta_{cur_beta}_iter_{iteration}')
    fig.tight_layout()
    fig.savefig(f'results/fields_{iteration}.png')

    plt.close('all')
    return cost

opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=[J1],
    objective_arguments=obj_list,
    design_regions=[mat_grid_region],
    frequencies=[freq0],
    maximum_run_time=2000,
    decay_by=1e-5
)

def f(v, gradient, cur_beta):
    global iteration, fmin
    iteration += 1 
    print(f'{"="*60}')
    print(f'Iteration = {iteration}')

    f0, dJ_du = opt([mapping(v, eta_i, cur_beta)])  # compute objective and gradient

    print(f'{f0=}')
    if f0 < fmin:
        print(f'{"*"*80}')
        print(f'New Fmin: {f0}')
        print(f'{"*"*80}')
        fmin = f0
        np.save('results/fmin.npy',v)

    if gradient.size > 0:
        gradient[:] = tensor_jacobian_product(mapping, 0)(
            v, eta_i, cur_beta, dJ_du
        )  # backprop

    print(f'{np.linalg.norm(gradient)=}')

    if mp.am_really_master():
        fig,ax = plt.subplots(2,1,figsize=(12,9))
        opt.plot2D(
            False,
            ax=ax[0],
            show_sources=False,
            show_monitors=False,
            show_boundaries=False,
        )
        ax[0].axis("off")
        
        im =ax[1].pcolormesh(np.reshape(gradient,(DR_N_H, DR_N_X)), cmap='bwr', shading='gouraud')
        fig.colorbar(im)
        fig.suptitle(f'beta_{cur_beta}_iter_{iteration}')
        fig.tight_layout()
        fig.savefig(f'results/params_iter{iteration}.png')
        plt.close('all')

        with open('results/opt_history.txt','ab') as f:
            np.savetxt(f, np.atleast_1d(getval(f0)))
    
    return np.real(f0)


algorithm = nlopt.LD_CCSAQ
n = DR_N_X * DR_N_H  # number of parameters

# Initial guess
x = np.ones((n,)) * 0.5

# lower and upper bounds
lb = np.zeros((DR_N_X * DR_N_H,))
ub = np.ones((DR_N_X * DR_N_H,))

RUN_OPT = True
if RUN_OPT:
    cur_beta = 1.2
    beta_scale = 2
    num_betas = 3
    update_factor = 40
    ftol = 1e-5
    print('Starting Optimization')
    for iters in range(num_betas):
        print(f'{"*"*80}')
        print(f'Current Beta: {cur_beta}')
        print(f'{"*"*80}')
        solver = nlopt.opt(algorithm, n)
        solver.set_lower_bounds(lb)
        solver.set_upper_bounds(ub)
        solver.set_min_objective(lambda a, g: f(a, g, cur_beta))
        solver.set_maxeval(update_factor)
        solver.set_ftol_rel(ftol)
        x[:] = solver.optimize(x)
        np.save(f'results/xmin_beta_{cur_beta}.npy',x)
        cur_beta = cur_beta * beta_scale


# --------------------------------------------------------------------------------------#
#                           Run Final 
# --------------------------------------------------------------------------------------#


mat_grid = mp.MaterialGrid(
    grid_size= mp.Vector3(DR_N_X,DR_N_H),
    medium1=mp.air,
    medium2=mp.Medium(index=n_radix),
    beta=4,
    eta=0.5,
    do_averaging=True,
    weights=x,
    grid_type="U_MEAN"
)
mat_grid_region = mpa.DesignRegion(
    mat_grid,
    volume=mp.Volume(
        center=mp.Vector3(0,0),
        size=DR_CM
    )
)
atoms = mp.Block(
    material=mat_grid,
    center=mat_grid_region.center,
    size=mat_grid_region.size
)

geometry = [atoms]

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    symmetries=symmetries,
                    split_chunks_evenly=False,
                    resolution=resolution)

RUN_FINAL = True
if RUN_FINAL:

    n2f_obj = sim.add_near2far(
        freq0, 0, 1,
        mp.Near2FarRegion(
            center=mp.Vector3(0,height/2 + buffer_z), 
            size=mp.Vector3(DRX,0), weight=+1
            )
        )

    if mp.am_really_master():
        fig,ax1=plt.subplots(1,1,figsize=(15,15))
        sim.plot2D(ax=ax1,output_plane=mp.Volume(center=mp.Vector3(),size=mp.Vector3(cell.x,cell.y)))
        fig.tight_layout()
        fig.savefig('results/meta_geo_final.png')

    sim.run(until_after_sources=mp.stop_when_energy_decayed(10, 1e-6))
    ffields = sim.get_farfields(n2f_obj, resolution=20, center=mp.Vector3(0,FF_DIST), size=mp.Vector3(10*lda0))
    ifar = np.abs(ffields['Ez'])**2

    xcfields = sim.get_farfields(n2f_obj, resolution=20, center=mp.Vector3(0,FF_DIST), size=mp.Vector3(10*lda0,0.5*FF_DIST))
    ixc = np.abs(xcfields['Ez'])**2
    if mp.am_really_master():
        fig, ax = plt.subplots(1,1,figsize=(5,5)) 
        ax.plot(ifar)
        fig.tight_layout()
        fig.savefig('results/metalens_farfield_final.png')

        fig, ax = plt.subplots(1,1,figsize=(5,5)) 
        ax.pcolormesh(ixc.T, cmap=paper_cmap, shading='gouraud')
        fig.tight_layout()
        fig.savefig('results/metalens_farfield_xc_final.png')
