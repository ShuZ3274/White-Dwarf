import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


def make_evo_matrix(dt, dx, x_grid, eta_val, l = 1,  leak_surf = False, leak_core = True):
    # dt, dx time step grid spacing
    # need to pass it grid space and array first, function won't construct it
    # initial magnetic profile to start with
    # eta_val is eta profile to generate matrix
    # l the mode index: 1,2,3...
    
    # Returns:
    # an operator matrix that evolve one time step for dt duration
    # using Crank-Nickleson method, with flexible boundary precision

    alpha = eta_val * dt / dx **2
    beta = dt * eta_val * l * (l+1) / x_grid **2
    impA = np.diag(-alpha[1:]/2, k=-1) + np.diag(-alpha[:-1]/2, k=1) + np.diag(1+ alpha + 0.5* beta)
    expB = np.diag(alpha[1:]/2, k=-1) + np.diag(alpha[:-1]/2, k=1) + np.diag(1-alpha - 0.5* beta)

    # Boundary condition:
    # at surface: df/dx = (l * f(x))/x 
    expB[-1][-1] = 1 - alpha[-1] * (1 + (dx * l) / x_grid[-1]) - 0.5 * beta[-1]
    impA[-1][-1] = 1 + alpha[-1] * (1 + (dx * l) / x_grid[-1]) + 0.5 * beta[-1]
    impA[-1][-2] = -alpha[-1]
    expB[-1][-2] = alpha[-1]
    # at core: df/dx = ((l + 1) * f(x))/x 
    expB[0][0] = 1 - alpha[0] * (1 + (dx * (l + 1)) / x_grid[0]) - 0.5 * beta[0]
    impA[0][0] = 1 + alpha[0] * (1 + (dx * (l + 1)) / x_grid[0]) + 0.5 * beta[0]
    impA[0][1] = -alpha[0]
    expB[0][1] = alpha[0]

    # leak BC to an extra layer on surface;
    if leak_surf == True:
        impA[-2][-2] = 1 + alpha[-2] * (1 + (dx * l) / x_grid[-2]) + 0.5 * beta[-2]
        impA[-2][-1] = 0.0
        expB[-2][-2] = 1 - alpha[-2] * (1 + (dx * l) / x_grid[-2]) - 0.5 * beta[-2]
        expB[-2][-1] = 0.0
        impA[-2][-3] = -alpha[-2]
        expB[-2][-3] = alpha[-2]
    # leak to core
    if leak_core == True:
        impA[1][1] = 1 + alpha[1] * (1 + (dx * (l + 1)) / x_grid[1]) + 0.5 * beta[1]
        impA[1][0] = 0.0
        expB[1][1] = 1 - alpha[1] * (1 + (dx * (l + 1)) / x_grid[1]) - 0.5 * beta[1]
        expB[1][0] = 0.0
        impA[1][2] = -alpha[1]
        expB[1][2] = alpha[1]
     
    imex = np.linalg.inv(impA) @ expB

    return imex



def freeze_the_core(eta_val, x_grid, mag = 1000, solidcore_size = 0.0):
    # input etaval and x_grid should have same size and to be discrete
    # input solidcore size could be size in cgs:
    # should input original eta_profile every time

    # Returns:
    # new eta value

    if solidcore_size >= max(x_grid):
        raise ValueError("Core size is larger than surface")
    
    size_index = int( len(x_grid) * solidcore_size / max(x_grid))
    new =  eta_val.copy()
    new[: size_index] =  new[: size_index] / mag
    return new 

def conv_boost(eta_val, x_grid, mag = 2000, solidcore_size = 0.0, conv_zone_size = 0.0):
    # input etaval and x_grid should have same size and to be discrete
    # input solidcore size could be size in cgs:
    # should input original eta_profile every time

    # Returns:
    # new eta value

    if solidcore_size >= conv_zone_size:
        # raise ValueError("Core size outgrows conv size")
        return eta_val.copy()
    
    core_ind = int( len(x_grid) * solidcore_size / max(x_grid))
    conv_ind = int( len(x_grid) * conv_zone_size / max(x_grid))
    new =  eta_val.copy()
    new[core_ind: conv_ind] =  new[core_ind: conv_ind] * mag
    return new 


def build_initial(x_grid, R_core, R_out, mag = 1):
    # take a range of R_out and core in cgs

    # Returns:
    # the B field profile

    R = np.zeros(len(x_grid))
    
    if R_out >= max(x_grid):
        raise ValueError("Convection region size exceeding surface")

    init_i = int( len(x_grid) * R_core / max(x_grid))
    fin_i = int( len(x_grid) * R_out / max(x_grid))

    R[init_i : fin_i] = mag

    return R * (x_grid / max(x_grid) ) **2 



def evo_R(R_0, eta, dt, nsteps, dx, x_grid):
    # initial B field profile R_0
    # eta profile
    # number of steps: nstep

    # Returns:
    # R_final: B field profile after nsteps

    # Store time evolution of field
    npoints = len(R_0)
    R_evo = np.zeros(shape=[nsteps+1, npoints])
    R_evo[0] = R_0
    r_past = np.zeros(shape=[nsteps, npoints])
     
    imex = make_evo_matrix(dt, dx, x_grid, eta)

    for i in range(nsteps):
        r_past[i] = imex @ R_evo[i]
        R_evo[i+1] = r_past[i]

    return R_evo


def evo_R_freeze(R_0, eta_origin, dt, evo_freq, dx, x_grid, free_freq, core):
    # evo_freq = nsteps
    # every dt * evo_freq / free_freq free the core
    # keep evolving for t_end /  free_freq
    # freeze, repeat

    # Returns:
    # R_final: B field profile after nsteps
    eta_ori = eta_origin.copy()

    # value storage:
    step = int (evo_freq / free_freq)
    npoints = len(R_0)
    R_evo = np.zeros(shape=[free_freq+1, npoints])
    R_evo[0] = R_0
    r_past = np.zeros(shape=[free_freq, npoints])


    for i in range(free_freq):
        eta =  freeze_the_core(eta_ori, x_grid, solidcore_size= core[i])
        r_past[i] = evo_R(R_evo[i], eta, dt, step, dx, x_grid)[-1]
        R_evo[i+1] = r_past[i]
        
    return R_evo



def convert_R_to_psi(origin_xgrid, Revo, extend = 0.5, size = 512):

    bound = (1 + extend) * max(origin_xgrid)

    x = np.linspace(- bound, bound, size)
    y = np.linspace(- bound, bound, size)

    xgrid, ygrid = np.meshgrid(x, y, indexing="ij")
    rgrid = np.sqrt (xgrid **2 + ygrid **2)
    theta_grid  = np.arctan2(ygrid, xgrid)


    # extend the field outside surface use boundary condition:
    # where B = B0 / 2r^2
    R_length = len(Revo)
    extend_factor = (1 + extend) * 1.45 
    extend_R = np.zeros(int(R_length * extend_factor))
    ind =  np.arange(int(R_length * extend_factor))


    r_extend = np.linspace(min(origin_xgrid), max(origin_xgrid)* extend_factor, int(R_length * extend_factor))

    extend_R[:R_length] = Revo
    B_0 = Revo[-1]
    extend_R[R_length :] = B_0 * (R_length / ind[R_length :]) 


    # interpolation
    f = interpolate.interp1d(r_extend, extend_R)
    interp_R = f(rgrid.flatten())
    R_inr = interp_R.reshape([size, size])

    # theta part
    leg_func = np.abs(np.sin(theta_grid))
    theta_part = leg_func * np.sin(theta_grid)




    return R_inr * theta_part, xgrid, ygrid
    


def r_Bessel (rho, index_l =  0):
    if index_l == 0:
        return np.sin(rho) / rho
    if index_l == 1:
        return np.sin(rho) / rho**2 - np.cos(rho) /rho




