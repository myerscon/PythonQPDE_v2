"""
Input file for heat equation PDE simulation.

This module defines the parameters and equations for solving the standard heat equation
using the quantum PDE solver. Demonstrates basic diffusion equation setup with fixed
boundary conditions and Gaussian initial conditions.

Equation: ∂T/∂t = α ∇²T

Where T is temperature, α is thermal diffusivity, and the spatial domain is 1D.
"""

import sympy as sp
import numpy as np

##########################
### GENERAL PARAMETERS ###
##########################
directory_name = 'problems/heat_equation/' # output directory name
verbose = True # Prints out status/diagnostics when true

# NUMERICAL PARAMETERS
taylor_r = 2
index_range = 1
integral_mode = 'quad' # integration scheme; "riemann" or "quad" supported
n_samples = 2 # number of samples for integration. Only 2 is needed for 4-th order accurate in time when using Gauss quadrature
use_vectorized = True
time_discretization =  'implicit_pick_nk' # 'implicit' # 'explicit' # implicit_pick_nk

# PROBLEM PARAMETERS
xlims = (0.,10.)
N = 101
h_value = (xlims[1]-xlims[0])/(N-1)
CFL = 0.8 # The timestep is constrained by instability at a larger CFL (we currently employ an explicit time integration scheme)
dt = CFL*(h_value**2)/2 # This expression comes the CFL condition ∆t < CFL*(∆x)**2/D
x_array = np.linspace(xlims[0], xlims[1], N)  # Use linspace for exact endpoint matching
ic_array = 100*np.sin((np.pi/2)*(x_array/10)) # 100*np.exp(x_array - 10) # 100*np.sin((np.pi/2)*(x_array/10))
initial_cond_type = ['specified']
initial_cond_vals = [ic_array]
boundary_type = [('fixed','reflecting')] # Dictates the boundary type at the left and right values
boundary_vals = [(0.0,100.0)] # Left and right boundary values (fixed) -> If not fixed, use 'None' or an arbitrary value (unused)
num_ODEs = 1
num_aux = 0
# Add the time_dependent_funcs to the general parameters
time_dependent_funcs = [[None, None]]  # Left boundary is time-dependent, right is None (open)

# SOLVER PARAMETERS
quantum = False # Boolean value indicating whether to run the simulation in quantum or classical configurations
epsilon_1 = 0.005   # upper bound on error of approximate value found for function mean using quantum amplitude estimation algorithm
delta = 0.001   # probability quantum integration result violates its error upper bound (See F. Gaitan. Circuit implementation of oracles
                # used in a quantum algorithm for solving nonlinear partial differential equations. Phys. Rev. A, 109:032604, 2024.)
N_tot = 40000 # Total subintervals
total_time = 100.0
time_outputs = [0.,0.2,0.4,0.6,0.8,1.0] # Custom output times specified by the user
time_outputs *= 100*np.ones_like(time_outputs)

# GENERAL PARAMETER DICTIONARY
general_param_dict = {'directory_name':directory_name,'verbose':verbose,'taylor_r':taylor_r,'index_range':index_range,'integral_mode':integral_mode,\
                      'xlims':xlims,'N':N,'dt':dt,'initial_cond_type':initial_cond_type,'initial_cond_vals':initial_cond_vals,'boundary_type':boundary_type,\
                      'boundary_vals':boundary_vals,'num_ODEs':num_ODEs,'num_aux':num_aux, 'quantum':quantum,'epsilon':epsilon_1,'N_tot':N_tot,\
                      'n_samples':n_samples,'delta':delta,'time_outputs':time_outputs,'use_vectorized':use_vectorized,'time_dependent_funcs':time_dependent_funcs,\
                      'time_discretization':time_discretization,'total_time':total_time}

########################
### SYMPY PARAMETERS ###
########################
# Constants
dz = sp.Symbol(r"\Delta z")
alpha = sp.Symbol(r"\alpha")
hbar = sp.Symbol(r"\bar{h}")

# Spatial and temporal variables
z = sp.Symbol("z")
t = sp.Symbol("t")
t_plus_hbar = sp.Symbol(r"t+\bar{h}")

# Variable functions
T = sp.Function("T")(z,t)

# Driver functions
f = alpha*(T.diff(z).diff(z))

# List the constants serially in a single list.
dz_value = (xlims[1]-xlims[0])/(N-1)
list_constants = [[dz, alpha, hbar],[dz_value,0.5,dt]]

# List the spatial variable and temporal variable. List the spatial variables together while keeping the temporal variable independent
list_spatial_vars = [z]
temporal_var = t

# List the equations (the number of variable/driver function pairs) for the auto_sympy script. Create a list of pairs of each variable and driver func
list_equations = [[T,f]]

# List the auxiliary functions. Create a list of pairs of each auxiliary function and its Sympy derivative expression (for substitution)
list_auxiliary = []

# Sympy Parameter Dictionary
sympy_param_dict = {'list_equations':list_equations,'list_auxiliary':list_auxiliary,'list_spatial_vars':list_spatial_vars,\
                    'list_constants':list_constants,'temporal_var':temporal_var}