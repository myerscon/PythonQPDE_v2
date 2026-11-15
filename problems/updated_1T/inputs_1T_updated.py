# Input file containing the equations and corresponding variables and constants to be solved

import sympy as sp
import numpy as np

##########################
### GENERAL PARAMETERS ###
##########################
directory_name = 'problems/updated_1T/' # output directory name
verbose = True # Prints out status/diagnostics when true

# NUMERICAL PARAMETERS
taylor_r = 2
index_range = 1  # Updated from 3 to 1 to match current version
integral_mode = 'quad' # integration scheme; "riemann" or "quad" supported
n_samples = 2 # number of samples for integration. Only 2 is needed for 4-th order accurate in time when using Gauss quadrature
use_vectorized = True
time_discretization = 'explicit'  # Change to 'implicit' or 'implicit_pick_nk' for implicit mode

# PROBLEM PARAMETERS
xlims = (0.,1.)
N = 101
h_value = (xlims[1]-xlims[0])/(N-1)
CFL = 0.0008 # The timestep is constrained by instability at a larger CFL (we currently employ an explicit time integration scheme)
dt = CFL*(h_value**2)*3 # This expression comes the CFL condition ∆t < CFL*(∆x)**2/D
x_array = np.linspace(xlims[0], xlims[1], N)  # Use linspace for exact endpoint matching
delta = 0.01
gaussian_array = 0.01+0.99*np.exp(-(x_array)/delta)
initial_cond_type = ['specified']
initial_cond_vals = [gaussian_array**(4)] # The material equation is in terms of T = temp**4 = E
boundary_type = [('fixed','open')] # Dictates the boundary type at the left and right values
boundary_vals = [(1.0,None)] # Left and right boundary values (fixed) -> If not fixed, use 'None' or an arbitrary value (unused)
num_ODEs = 1
num_aux = 1
# Add the time_dependent_funcs to the general parameters
time_dependent_funcs = [[None, None]]  # No time-dependent boundaries for this problem

# SOLVER PARAMETERS
quantum = False # Boolean value indicating whether to run the simulation in quantum or classical configurations
epsilon_1 = 0.005   # upper bound on error of approximate value found for function mean using quantum amplitude estimation algorithm
delta = 0.001   # probability quantum integration result violates its error upper bound (See F. Gaitan. Circuit implementation of oracles
                # used in a quantum algorithm for solving nonlinear partial differential equations. Phys. Rev. A, 109:032604, 2024.)
N_tot = 4500000 # Total subintervals
total_time = 1.08  # For implicit mode - covers the range of time_outputs
#time_outputs = [0.0,0.005,0.01,0.02,0.04,0.08,0.12,0.16,0.20,0.24,0.28,0.32,0.36,0.54,0.72,0.9,1.08] # Custom output times specified by the user
time_outputs = [0.0,0.005,0.007,0.01,0.02,0.04,0.046,0.08,0.126,0.251,0.377,0.503,0.628,0.754,0.88]

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
gamma = sp.Symbol(r"\gamma")
hbar = sp.Symbol(r"\bar{h}")

# Spatial and temporal variables
z = sp.Symbol("z")
t = sp.Symbol("t")
#t_plus_hbar = sp.Symbol(r"t+\bar{h}")

# Variable functions
E = sp.Function("E")(z,t)

# Auxiliary functions - optional functions used to simplify expressions, coupling, etc
D = sp.Function("D")(z,t)

# Driver functions
f = (D*E.diff(z)).diff(z)

# Derivative of auxiliary functions - needed to substitute d(aux)/dt in Sympy expressions
D_expr = (E**(gamma/4))/3
D_deriv = (gamma/4)*sp.Derivative(E,t)*(D/E)

# List the constants serially in a single list.
dz_value = (xlims[1]-xlims[0])/(N-1)
list_constants = [[dz, gamma, hbar],[dz_value,3.5,dt]]

# List the spatial variable and temporal variable. List the spatial variables together while keeping the temporal variable independent
list_spatial_vars = [z]
temporal_var = t

# List the equations (the number of variable/driver function pairs) for the auto_sympy script. Create a list of pairs of each variable and driver func
list_equations = [[E,f]]

# List the auxiliary functions. Create a list of pairs of each auxiliary function and its Sympy derivative expression (for substitution)
list_auxiliary = [[D,D_expr,D_deriv]]

# Sympy Parameter Dictionary
sympy_param_dict = {'list_equations':list_equations,'list_auxiliary':list_auxiliary,'list_spatial_vars':list_spatial_vars,\
                    'list_constants':list_constants,'temporal_var':temporal_var}