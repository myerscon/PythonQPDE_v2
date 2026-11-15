# Input file containing the equations and corresponding variables and constants to be solved

import sympy as sp
import numpy as np

##########################
### GENERAL PARAMETERS ###
##########################
directory_name = 'problems/test_2T/' # output directory name
verbose = True # Prints out status/diagnostics when true

# NUMERICAL PARAMETERS
taylor_r = 2
index_range = 1
integral_mode = 'quad' # integration scheme; "riemann" or "quad" supported
n_samples = 2 # number of samples for integration. Only 2 is needed for 4-th order accurate in time when using Gauss quadrature
use_vectorized = True
time_discretization = 'implicit_pick_nk' # 'implicit_pick_nk' # 'implicit' # 'explicit'

# PROBLEM PARAMETERS
xlims = (0.,2.01)
N = 201
h_value = (xlims[1]-xlims[0])/(N-1)
CFL = 0.01 # The timestep is constrained by instability at a larger CFL (we currently employ an explicit time integration scheme)
dt = CFL*(h_value**2)*3 # This expression comes the CFL condition ∆t < CFL*(∆x)**2/D
initial_cond_type = ['specified','specified']
x_array = np.arange(0,2.01,0.01)
sigma = 0.01
gaussian_array = 0.01+0.99*np.exp(-(x_array)/sigma)
initial_cond_vals = np.array([[gaussian_array],[gaussian_array]]) # The material equation is in terms of T = temp**2 = sqrt(E)
# ADDED: TIME DEPENDENT BOUNDARIES
def time_dependent_left(t,r):
    return np.exp(-t/0.2)
    
#boundary_type = [('fixed','open'),('fixed','open')] # Dictates the boundary type at the left and right values -> 'fixed', 'open'
#boundary_vals = [(1.,None),(1,None)] # Left and right boundary values (fixed) -> If not fixed, use 'None' or an arbitrary value (unused)
boundary_type = [('fixed','fixed'),('fixed','fixed')] # Dictates the boundary type at the left and right values -> 'fixed', 'open'
boundary_vals = [(1,0.01),(1,0.01)] # Left and right boundary values (fixed) -> If not fixed, use 'None' or an arbitrary value (unused)
num_ODEs = 2 # Potentially remove in the future, rely on auto_sympy for this value
num_aux = 1 # Potentially remove in the future, rely on auto_sympy for this value
# Add the time_dependent_funcs to the general parameters
time_dependent_funcs = [[time_dependent_left, None],[None,None]]  # Left boundary is time-dependent, right is None (open)

# SOLVER PARAMETERS
quantum = False # Boolean value indicating whether to run the simulation in quantum or classical configurations
epsilon_1 = 0.005   # upper bound on error of approximate value found for function mean using quantum amplitude estimation algorithm
delta = 0.001   # probability quantum integration result violates its error upper bound (See F. Gaitan. Circuit implementation of oracles
                # used in a quantum algorithm for solving nonlinear partial differential equations. Phys. Rev. A, 109:032604, 2024.)
N_tot = 600000 # Total subintervals
total_time = 1.8 # 0.45 1.8
#time_outputs = 4*np.ones(18)*[0.0,0.0125,0.025,0.0375,0.05,0.0625,0.075,0.0875,0.1,0.125,0.15,0.175,0.2,0.25,0.3,0.35,0.4,0.45]
time_outputs = 16*np.ones(13)*[0.0,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]

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
dz = sp.Symbol("dz")
gamma = sp.Symbol("gamma")
hbar = sp.Symbol(r"\bar{h}")
c_ = sp.Symbol("c_")

# Spatial and temporal variables
z = sp.Symbol("z")
t = sp.Symbol("t")

# Variable functions
#     -Each variable function should have a corresponding driver function
#     -The driver function and it's derivatives up to order r are used to approximate the solution
E = sp.Function("E")(z,t)
T = sp.Function("T")(z,t)

# Auxiliary functions - optional functions used to simplify expressions, coupling, etc
D = sp.Function("D")(z,t)

# Driver functions
f = (D*E.diff(z)).diff(z) + (T-E)/(T**(gamma/4))

# Derivative of auxiliary functions - needed to substitute d(aux)/dt in Sympy expressions
g = -(T-E)/(c_*(T**(gamma/4))) # multiplication factor of E in denominator is either 5.0 or 500.0

D_expr = (T**(gamma/4))/3
D_deriv = gamma*(D/T)*sp.Derivative(T, t)

# List the constants serially in a single list.
# dz should always be the first constant. It is calculated from xlims and N below.
dz_value = (xlims[1]-xlims[0])/(N-1)
list_constants = ([dz,gamma,hbar,c_],[dz_value,5.6,dt,0.2]) # [dz_value,3.53,3.53,156.4] [dz_value,5.6,5.6,2.5] [dz_value,3.53,3.53,0.415288]

# List the spatial variable and temporal variable. List the spatial variables together while keeping the temporal variable independent
list_spatial_vars = [z]
temporal_var = t

# List the equations (the number of variable/driver function pairs) for the auto_sympy script. Create a list of pairs of each variable and driver func
list_equations = [[E,f],[T,g]]

# List the auxiliary functions. Create a list of pairs of each auxiliary function and its Sympy derivative expression (for substitution)
list_auxiliary = [[D,D_expr,D_deriv]]

# Sympy Parameter Dictionary
sympy_param_dict = {'list_equations':list_equations,'list_auxiliary':list_auxiliary,'list_spatial_vars':list_spatial_vars,\
                    'list_constants':list_constants,'temporal_var':temporal_var}