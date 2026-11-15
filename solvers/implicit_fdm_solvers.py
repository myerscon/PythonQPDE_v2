"""
Implicit finite difference method solver implementations.

This module provides implicit time stepping schemes for solving partial differential
equations using finite difference spatial discretization. Includes both non-vectorized
and vectorized implementations that solve tridiagonal matrix systems at each time step.

The implicit solvers use backward time stepping where the solution at the next time
step requires solving a system of equations coupling spatial grid points.
"""

import numpy as np
import kacewicz_functions
from solvers.simulation import Simulation
from sympy_files.fdm_equation_writer import Implicit_fdm_equation_writer


def _point_coeffs(sim, C_all, i, n_int):
    """Concatenate the Taylor coefficients of ALL equations at points i-1, i, i+1 (zeros outside the grid)."""
    zero = np.zeros(sim.r + 2)
    parts = []
    for e in range(C_all.shape[0]):
        parts.append(C_all[e, :, i - 1] if i - 1 >= 0 else zero)
        parts.append(C_all[e, :, i])
        parts.append(C_all[e, :, i + 1] if i + 1 < n_int else zero)
    return np.concatenate(parts)


def _boundary_arrays(sim, t_old):
    """Per-equation boundary alpha / alpha_dot arrays: (left, left_dot, right, right_dot)."""
    vals = [sim._get_primed_boundary_values(e, t_old) for e in range(sim.num_ODEs)]
    return (np.array([v['left'][0] for v in vals]), np.array([v['left'][1] for v in vals]),
            np.array([v['right'][0] for v in vals]), np.array([v['right'][1] for v in vals]))


class Implicit_fdm_simulation(Simulation):
    """ Non-vectorized implicit finite difference simulation.
    
    Implements implicit time stepping that requires solving tridiagonal matrix systems
    to advance the solution. Uses Thomas algorithm for efficient tridiagonal solving
    and handles primed integral corrections for improved accuracy.
    
    The implicit scheme couples spatial grid points through a linear system that must
    be solved at each time step, enabling different stability properties compared to
    explicit methods.
    
    Attributes:
        Inherits all attributes from Simulation base class.
        Additional wrapper functions for primed integral evaluation.
        
    Notes:
        Requires more computation per time step than explicit methods due to matrix
        solving. Handles custom output times by solving additional matrix systems
        for intermediate time values.
    """    
    def __init__(self, general_params, sympy_params):
        """ Initialize implicit finite difference simulation.
        
        Creates wrapper functions for primed integral evaluation. Prepares integrand 
        functions for quantum integration of correction terms specific to the implicit
        scheme.
        
        Args:
            general_params (dict): Simulation parameters including grid, time stepping,
                boundary conditions, and solver options.
            sympy_params (dict): Symbolic equation definitions, auxiliary functions,
                spatial variables, and constants.
        """
        super().__init__(general_params, sympy_params)
        
        # Add these attributes to store current integrand parameters
        self._current_a_lambda = None
        self._current_b_lambda = None
        self._current_c_lambda = None
        self._current_coeffs = None
        self._current_alpha_left = 0.0
        self._current_alpha_right = 0.0
        self._current_alpha_dot_left = 0.0
        self._current_alpha_dot_right = 0.0
        
        # Create wrapper functions once
        def a_integrand_func(t, t0, C_dummy):
            return self._current_a_lambda(t, t0, *self._current_coeffs, 
                                        self._current_alpha_left, self._current_alpha_right, 
                                        self._current_alpha_dot_left, self._current_alpha_dot_right)
            
        def b_integrand_func(t, t0, C_dummy):
            return self._current_b_lambda(t, t0, *self._current_coeffs, 
                                        self._current_alpha_left, self._current_alpha_right, 
                                        self._current_alpha_dot_left, self._current_alpha_dot_right)
            
        def c_integrand_func(t, t0, C_dummy):
            return self._current_c_lambda(t, t0, *self._current_coeffs, 
                                        self._current_alpha_left, self._current_alpha_right, 
                                        self._current_alpha_dot_left, self._current_alpha_dot_right)
        
        # Store these functions as instance methods
        self._a_integrand_func = a_integrand_func
        self._b_integrand_func = b_integrand_func
        self._c_integrand_func = c_integrand_func

        print("Created wrapper functions for implicit simulation")

    def _initialize_auto_sympy(self, sympy_params):
        """ Initialize symbolic equation processing for implicit scheme.
        
        Sets up the equation writer and performs implicit finite difference approximation
        including tridiagonal matrix coefficient generation and primed integral setup.
        
        Args:
            sympy_params (dict): Symbolic parameters including equations, auxiliaries,
                spatial variables, and constants.
                
        Notes:
            Uses Implicit_fdm_equation_writer to create tridiagonal coefficients and
            primed integral expressions needed for implicit time stepping accuracy.
        """
        self.evals = Implicit_fdm_equation_writer(
            sympy_params['list_equations'],
            sympy_params['list_auxiliary'],
            sympy_params['list_spatial_vars'],
            sympy_params['list_constants'],
            sympy_params['temporal_var'],
            self.r,
            self.index_range,
            self.factorials
        )
        self.evals.implicit_finite_diff_approx(
            z=sympy_params['list_spatial_vars'][0],     # spatial variable
            dz_sym=sympy_params['list_constants'][0][0] # grid spacing symbol
        )

    @staticmethod
    def solve_tridiagonal(a, b, c, d):
        """ Solve tridiagonal system using Thomas algorithm.
        
        Efficiently solves the linear system Ax = d where A is tridiagonal with
        lower diagonal a, main diagonal b, and upper diagonal c.
        
        Args:
            a (np.ndarray): Lower diagonal elements (length n-1).
            b (np.ndarray): Main diagonal elements (length n).
            c (np.ndarray): Upper diagonal elements (length n-1).
            d (np.ndarray): Right-hand side vector (length n).
            
        Returns:
            np.ndarray: Solution vector x (length n).
            
        Notes:
            Uses forward elimination followed by back substitution. More efficient
            than general matrix solvers for tridiagonal systems.
        """
        n = len(d)
        
        # Declare buffer and output arrays
        c_prime = np.empty(n-1)
        d_prime = np.empty(n)
        x = np.empty(n)

        # Forward sweep - sequential
        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]
        
        for i in range(1, n-1):
            denominator = b[i] - a[i-1] * c_prime[i-1]
            c_prime[i] = c[i] / denominator
            d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / denominator
        
        # Last equation
        d_prime[n-1] = (d[n-1] - a[n-2] * d_prime[n-2]) / (b[n-1] - a[n-2] * c_prime[n-2])
        
        # Back substitution - sequential
        x[n-1] = d_prime[n-1]
        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]
            
        return x
    
    def get_lambda_inputs(self, gridpoint_indices):
        """ Construct inputs for lambda functions based on equation writer requirements.
        
        Creates consistently ordered argument list for evaluating tridiagonal matrix
        coefficients and other lambda functions in the implicit scheme.
        
        Args:
            gridpoint_indices (np.ndarray): Array of interior grid point indices.
            
        Returns:
            list: Ordered arguments for lambda function evaluation including index,
                spatial variables, auxiliaries, and equation derivatives.
                
        Notes:
            Order: index j, spatial variables, auxiliaries, equations and derivatives.
            Consistent ordering ensures all lambda functions use same argument structure.
        """
        # Pre-allocate list with known size
        total_size = 1 + len(self.evals.spatial_vars_list) + self.num_aux + self.num_ODEs * (1 + self.r)
        inputs = [None] * total_size
        
        idx = 0
        inputs[idx] = gridpoint_indices
        idx += 1
        
        # Spatial variables
        for var in range(len(self.evals.spatial_vars_list)):
            inputs[idx] = self.r_values
            idx += 1
            
        # Auxiliary values
        for aux in range(self.num_aux):
            inputs[idx] = self.aux[aux]
            idx += 1
            
        # Equations and derivatives
        for eqn in range(self.num_ODEs):
            inputs[idx] = self.U[eqn]
            idx += 1
            for tylr in range(self.r):
                inputs[idx] = self.f_array[eqn, tylr]
                idx += 1
                
        return inputs

    def compute_boundary_contribution(self, eqn, boundary_side, t_old, t_new):
        """ Compute boundary contributions based on boundary condition type.
        
        Calculates additional terms needed for the implicit scheme when boundary
        conditions are not simply fixed values.
        
        Args:
            eqn (int): Equation index.
            boundary_side (str): 'left' or 'right' boundary.
            t_old (float): Start time of integration interval.
            t_new (float): End time of integration interval.
            
        Returns:
            float: Boundary contribution to the right-hand side.
            
        Notes:
            Currently implements fixed boundary handling. Other boundary types
            return zero but are structured for future implementation.
        """
        side_index = 0 if boundary_side == 'left' else 1
        boundary_type = self.boundary_type[eqn][side_index]
        
        if boundary_type == 'fixed':
            return 0.0
        elif boundary_type == 'neumann':
            return 0.0  # To be implemented later
        elif boundary_type == 'open':
            return 0.0  # To be implemented later
        elif boundary_type == 'reflecting':
            return 0.0  # No additional contribution needed?
        elif boundary_type == 'time_dependent':
            return 0.0  # To be implemented later
        else:
            raise ValueError(f"Unknown boundary type: {boundary_type}")
    
    def _get_primed_boundary_values(self, eqn, t_old):
        """ Pre-compute boundary values to avoid repeated conditionals.
        
        Evaluates boundary conditions at the specified time to provide alpha
        and alpha_dot values needed for primed integral computation.
        
        Args:
            eqn (int): Equation index.
            t_old (float): Time for boundary value evaluation.
            
        Returns:
            dict: Boundary values with 'left' and 'right' keys containing
                (alpha, alpha_dot) tuples.
                
        Notes:
            Handles fixed, time-dependent, and other boundary types by evaluating
            appropriate expressions or using current solution values.
        """
        boundary_vals = {'left': (0.0, 0.0), 'right': (0.0, 0.0)}
        
        # Left boundary
        if self.boundary_type[eqn][0] == 'fixed':
            boundary_vals['left'] = (self.boundary_vals[eqn][0], 0.0)
        elif self.boundary_type[eqn][0] == 'reflecting':
            boundary_vals['left'] = (self.U[eqn, 1], 0.0)  # Use adjacent interior point
        elif self.boundary_type[eqn][0] == 'time_dependent':
            alpha_left = self.time_dependent_funcs[eqn][0](t_old, self.r_values[0])
            boundary_vals['left'] = (alpha_left, 0.0)
        else:
            boundary_vals['left'] = (self.U[eqn, 0], 0.0)
        
        # Right boundary  
        if self.boundary_type[eqn][1] == 'fixed':
            boundary_vals['right'] = (self.boundary_vals[eqn][1], 0.0)
        elif self.boundary_type[eqn][1] == 'reflecting':
            boundary_vals['right'] = (self.U[eqn, -2], 0.0)  # Use adjacent interior point
        elif self.boundary_type[eqn][1] == 'time_dependent':
            alpha_right = self.time_dependent_funcs[eqn][1](t_old, self.r_values[-1])
            boundary_vals['right'] = (alpha_right, 0.0)
        else:
            boundary_vals['right'] = (self.U[eqn, -1], 0.0)
            
        return boundary_vals

    def _assemble_matrix_coefficients(self, gridpoint_indices, eqn):
        """ Assemble tridiagonal matrix coefficients using equation writer outputs.
        
        Evaluates lambda functions to compute the a, b, c coefficients for the
        tridiagonal system corresponding to the specified equation.
        
        Args:
            gridpoint_indices (np.ndarray): Interior point indices.
            eqn (int): Equation index for coefficient computation.
            
        Returns:
            tuple: (a_, b_, c_) arrays for tridiagonal system.
            
        Notes:
            Coefficients depend on current solution state and auxiliary values.
            Used for both final solve and custom output time calculations.
        """
        lambda_inputs = self.get_lambda_inputs(gridpoint_indices)
        
        # Call all three lambda functions at once to potentially reuse computations
        eqn_str = str(eqn)
        a_key = 'a_' + eqn_str + '_lambdified'
        b_key = 'b_' + eqn_str + '_lambdified'  
        c_key = 'c_' + eqn_str + '_lambdified'
        
        # Cache the dict lookup
        c_dict = self.evals.c_dict
        
        a_ = c_dict[a_key](*lambda_inputs)
        b_ = c_dict[b_key](*lambda_inputs)
        c_ = c_dict[c_key](*lambda_inputs)
        
        return a_, b_, c_

    def _compute_rhs(self, y_current, I_i, a_, b_, c_):
        """ Compute right-hand side for tridiagonal system using vectorized operations.
        
        Builds the right-hand side vector by combining current solution state,
        integrated terms, and matrix-vector multiplication for the implicit system.
        
        Args:
            y_current (np.ndarray): Current solution state at interior points.
            I_i (np.ndarray): Accumulated integration results.
            a_ (np.ndarray): Lower diagonal coefficients.
            b_ (np.ndarray): Main diagonal coefficients.
            c_ (np.ndarray): Upper diagonal coefficients.
            
        Returns:
            np.ndarray: Right-hand side vector for tridiagonal solve.
            
        Notes:
            Implements: M * y_current + I_i where M is the tridiagonal matrix.
            Uses vectorized operations for efficiency.
        """
        
        # Vectorized matrix multiplication
        rhs = b_ * y_current + I_i  # Diagonal terms + integration
        
        # Upper diagonal terms: c[:-1] * y[1:]
        rhs[:-1] += c_[:-1] * y_current[1:]
        
        # Lower diagonal terms: a[1:] * y[:-1]  
        rhs[1:] += a_[1:] * y_current[:-1]
        
        return rhs

    def _integrate_secondary(self, t_old, t_new, eqn, secondary, gridpoint_indices):
        """ Perform integration for one secondary interval.
        
        Integrates the main driver function terms over the specified secondary
        interval for all interior spatial points of the given equation.
        
        Args:
            t_old (float): Start time of secondary interval.
            t_new (float): End time of secondary interval.
            eqn (int): Index of current equation.
            secondary (int): Index of current secondary interval.
            gridpoint_indices (np.ndarray): Array of interior point indices.
            
        Returns:
            np.ndarray: Integration results for interior points.
            
        Notes:
            Uses stored Taylor polynomial coefficients from the ll array.
            Processes each spatial point individually in non-vectorized approach.
        """
        N_interior = len(gridpoint_indices)
        results = np.empty(N_interior)  # We overwrite everything
        
        # Pre-compute constant values
        tlims_array = np.array([t_old, t_new])  # Create once
        C_all = self.ll[eqn, 1:, :, secondary]  # Extract once: shape (r+1, N-2)
        
        for i in range(N_interior):
            # Get coefficients for current point (just indexing, no .T needed)
            C = C_all[:, i]
            
            results[i] = kacewicz_functions.integrate(
                integrand_function=self.integrand_func,
                tlims=tlims_array,  # Reuse same array
                n_samples=self.n_samples,
                integral_mode=self.integral_mode,
                C=C,
                t0=t_old,
                quantum=self.quantum,
                M=self.M,
                delta=self.delta,
                use_vectorized=False
            )
            
        return results

    def _compute_primed_integrals(self, t_old, t_new, eqn, secondary, gridpoint_indices):
        """ Compute primed integrals with vectorized middle points.
        
        Evaluates the primed integral correction terms needed for implicit scheme
        accuracy. Handles boundary points individually and vectorizes interior points
        for computational efficiency.
        
        Args:
            t_old (float): Start time of secondary interval.
            t_new (float): End time of secondary interval.
            eqn (int): Index of current equation.
            secondary (int): Index of current secondary interval.
            gridpoint_indices (np.ndarray): Array of interior point indices.
            
        Returns:
            tuple: (I_1, I_2, I_3) arrays of primed integral contributions.
            
        Notes:
            Boundary points (i=0, i=N-1) handled individually due to boundary conditions.
            Middle points processed with vectorized operations for better performance.
            Uses quantum integration for all primed integral evaluations.
        """
        N_interior = len(gridpoint_indices)
        
        # Pre-allocate results
        I_1 = np.zeros(N_interior)
        I_2 = np.zeros(N_interior) 
        I_3 = np.zeros(N_interior)
        
        # Cache dictionary lookups
        eqn_str = str(eqn)
        c_dict = self.evals.c_dict
        lambda_funcs = {
            'left': (c_dict[f'a_{eqn_str}_prime_integrand_lambda_left'],
                    c_dict[f'b_{eqn_str}_prime_integrand_lambda_left'],
                    c_dict[f'c_{eqn_str}_prime_integrand_lambda_left']),
            'middle': (c_dict[f'a_{eqn_str}_prime_integrand_lambda_middle'],
                    c_dict[f'b_{eqn_str}_prime_integrand_lambda_middle'],
                    c_dict[f'c_{eqn_str}_prime_integrand_lambda_middle']),
            'right': (c_dict[f'a_{eqn_str}_prime_integrand_lambda_right'],
                    c_dict[f'b_{eqn_str}_prime_integrand_lambda_right'],
                    c_dict[f'c_{eqn_str}_prime_integrand_lambda_right'])
        }
        
        # Get coefficient array once
        C_all = self.ll[:, :, :, secondary]
        bL, bLd, bR, bRd = _boundary_arrays(self, t_old)
        zeros = np.zeros(self.num_ODEs)
        dummy_C = np.ones(1)
        
        # === VECTORIZED MIDDLE POINTS ===
        if N_interior > 2:  # Only if we have middle points
            middle_indices = np.arange(1, N_interior - 1)
            n_middle = len(middle_indices)
            
            if n_middle > 0:
                # Prepare vectorized coefficient arrays for middle points
                middle_coeffs = np.zeros((n_middle, 3 * self.num_ODEs * (self.r + 2)))
                
                for idx, i in enumerate(middle_indices):
                    middle_coeffs[idx] = _point_coeffs(self, C_all, i, N_interior)
                
                # Get middle lambda functions
                a_lambda, b_lambda, c_lambda = lambda_funcs['middle']
                
                # Vectorized integration for middle points
                for idx, i in enumerate(middle_indices):
                    # Set up for this middle point (no boundary values needed)
                    self._current_a_lambda = a_lambda
                    self._current_b_lambda = b_lambda
                    self._current_c_lambda = c_lambda
                    self._current_coeffs = middle_coeffs[idx]
                    self._current_alpha_left = 0.0
                    self._current_alpha_right = 0.0
                    self._current_alpha_dot_left = 0.0
                    self._current_alpha_dot_right = 0.0
                    
                    # Quantum integrations
                    I_1[i] = kacewicz_functions.integrate(
                        integrand_function=self._a_integrand_func,
                        tlims=np.array([t_old, t_new]),
                        n_samples=self.n_samples,
                        integral_mode=self.integral_mode,
                        C=dummy_C,
                        t0=t_old,
                        quantum=self.quantum,
                        M=self.M,
                        delta=self.delta,
                        use_vectorized=False
                    )
                    
                    I_2[i] = kacewicz_functions.integrate(
                        integrand_function=self._b_integrand_func,
                        tlims=np.array([t_old, t_new]),
                        n_samples=self.n_samples,
                        integral_mode=self.integral_mode,
                        C=dummy_C,
                        t0=t_old,
                        quantum=self.quantum,
                        M=self.M,
                        delta=self.delta,
                        use_vectorized=False
                    )
                    
                    I_3[i] = kacewicz_functions.integrate(
                        integrand_function=self._c_integrand_func,
                        tlims=np.array([t_old, t_new]),
                        n_samples=self.n_samples,
                        integral_mode=self.integral_mode,
                        C=dummy_C,
                        t0=t_old,
                        quantum=self.quantum,
                        M=self.M,
                        delta=self.delta,
                        use_vectorized=False
                    )
        
        # === BOUNDARY POINTS (i=0 and i=N_interior-1) ===
        for i in [0, N_interior - 1]:
            if i == 0:  # Left boundary point
                a_lambda, b_lambda, c_lambda = lambda_funcs['left']
                alpha_left, alpha_dot_left = bL, bLd
                alpha_right = alpha_dot_right = zeros
                
                # Coefficient handling for left boundary
                
            else:  # Right boundary point (i == N_interior - 1)
                a_lambda, b_lambda, c_lambda = lambda_funcs['right']
                alpha_left = alpha_dot_left = zeros
                alpha_right, alpha_dot_right = bR, bRd
                
                # Coefficient handling for right boundary
            
            # Concatenate coefficients
            all_coeffs = _point_coeffs(self, C_all, i, N_interior)
            
            # Set up boundary point
            self._current_a_lambda = a_lambda
            self._current_b_lambda = b_lambda
            self._current_c_lambda = c_lambda
            self._current_coeffs = all_coeffs
            self._current_alpha_left = alpha_left
            self._current_alpha_right = alpha_right
            self._current_alpha_dot_left = alpha_dot_left
            self._current_alpha_dot_right = alpha_dot_right
            
            # Quantum integrations for boundary point
            I_1[i] = kacewicz_functions.integrate(
                integrand_function=self._a_integrand_func,
                tlims=np.array([t_old, t_new]),
                n_samples=self.n_samples,
                integral_mode=self.integral_mode,
                C=dummy_C,
                t0=t_old,
                quantum=self.quantum,
                M=self.M,
                delta=self.delta,
                use_vectorized=False
            )
            
            I_2[i] = kacewicz_functions.integrate(
                integrand_function=self._b_integrand_func,
                tlims=np.array([t_old, t_new]),
                n_samples=self.n_samples,
                integral_mode=self.integral_mode,
                C=dummy_C,
                t0=t_old,
                quantum=self.quantum,
                M=self.M,
                delta=self.delta,
                use_vectorized=False
            )
            
            I_3[i] = kacewicz_functions.integrate(
                integrand_function=self._c_integrand_func,
                tlims=np.array([t_old, t_new]),
                n_samples=self.n_samples,
                integral_mode=self.integral_mode,
                C=dummy_C,
                t0=t_old,
                quantum=self.quantum,
                M=self.M,
                delta=self.delta,
                use_vectorized=False
            )

        return I_1, I_2, I_3

    def _handle_custom_output(self, t_old, eqn, secondary, y_current, custom_time, gridpoint_indices):
        """ Handle computation and storage of solution at custom output times.
        
        Computes solution at user-specified output times by integrating up to the
        custom time and solving the resulting tridiagonal system.
        
        Args:
            t_old (float): Start time of secondary interval.
            eqn (int): Index of current equation.
            secondary (int): Index of current secondary interval.
            y_current (np.ndarray): Current solution state.
            custom_time (float): Requested output time.
            gridpoint_indices (np.ndarray): Array of interior point indices.
            
        Notes:
            If custom_time equals t_old, uses current solution directly.
            Otherwise performs partial integration and matrix solve.
            Updates custom output arrays and saves to HDF5 file.
        """
        N_interior = len(gridpoint_indices)
        
        if custom_time == t_old:
            # At start of interval, use current solution
            self.custom_outputs[eqn, 1:-1, self.t_counter] = y_current
        else:
            # Pre-allocate and pre-compute constants
            custom_integration = np.empty(N_interior)
            custom_tlims = np.array([t_old, custom_time])
            C_all = self.ll[eqn, 1:, :, secondary]
            
            # Integrate up to custom time
            for i in range(N_interior):
                C = C_all[:, i]
                custom_integration[i] = kacewicz_functions.integrate(
                    integrand_function=self.integrand_func,
                    tlims=custom_tlims,
                    n_samples=self.n_samples,
                    integral_mode=self.integral_mode,
                    C=C,
                    t0=t_old,
                    quantum=self.quantum,
                    M=self.M,
                    delta=self.delta,
                    use_vectorized=False
                )
            
            # Compute primed integrals up to custom time
            I_1, I_2, I_3 = self._compute_primed_integrals(
                t_old, custom_time, eqn, secondary, 
                gridpoint_indices
            )
            
            # Combine all integral contributions
            total_integration = custom_integration + I_1 + I_2 + I_3
            
            # Get matrix coefficients and solve system
            a_, b_, c_ = self._assemble_matrix_coefficients(gridpoint_indices, eqn)
            rhs = self._compute_rhs(y_current, total_integration, a_, b_, c_)
            custom_solution = self.solve_tridiagonal(a_[1:], b_, c_[:-1], rhs)
            
            self.custom_outputs[eqn, 1:-1, self.t_counter] = custom_solution
        
        # If this is the last equation, handle boundaries and update counter
        if eqn == self.num_ODEs - 1:
            for i in range(self.num_ODEs):
                self.set_boundary_values(
                    self.custom_outputs, 
                    i, 
                    time_index=self.t_counter, 
                    current_time=custom_time
                )
            self.update_save_custom_h5(self.t_counter)
            self.t_counter += 1

    def IntegrateGij(self, t_start, primary_interval):
        """ Implicit integration scheme with matrix solve at end of primary intervals.
        
        Performs integration over all secondary intervals within a primary interval,
        accumulating integral terms and solving tridiagonal systems to advance the
        solution. Handles custom output times through additional matrix solves.
        
        Args:
            t_start (float): Starting time of the current primary interval.
            primary_interval (int): Index of the current primary interval.
            
        Notes:
            Accumulates three types of integrals: main driver function terms and
            primed correction terms I_1, I_2, I_3. Solves one tridiagonal system
            per equation at the end of each primary interval.
        """
        # Initialize arrays
        N_interior = self.N - 2
        gridpoint_indices = np.arange(1, self.N-1)
        
        # Current solution state
        y_current = self.y[:, 1:-1, primary_interval].copy()
        
        # Pre-compute time step and secondary count
        num_secondaries = self.n**(self.k-1)
        
        # Loop over each equation in the system
        for eqn in range(self.num_ODEs):
            # Pre-allocate accumulator arrays for all integral terms
            I_i = np.zeros(N_interior)
            I_1 = np.zeros(N_interior)
            I_2 = np.zeros(N_interior)
            I_3 = np.zeros(N_interior)
            
            # Accumulate integrations over all secondaries in this primary interval
            for secondary in range(num_secondaries):
                t_old = t_start + (secondary * self.hbar)
                t_new = t_start + ((secondary + 1) * self.hbar)
                
                # Handle custom output times if needed
                if self.t_counter < self.t_output_length:
                    custom_time = self.t_outputs[self.t_counter]
                    
                    # Add tolerance for floating point comparison
                    tolerance = 1e-10 * max(abs(custom_time), abs(t_new))
                    
                    if t_old <= custom_time < t_new or abs(custom_time - t_new) < tolerance:
                        self._handle_custom_output(
                            t_old, eqn, secondary,
                            y_current[eqn], custom_time, gridpoint_indices
                        )
                
                # Integrate over this secondary interval for main right-hand side
                I_i += self._integrate_secondary(
                    t_old, t_new, eqn, secondary, gridpoint_indices
                )
                
                # Compute primed integrals and update accumulator arrays
                I_1_current, I_2_current, I_3_current = self._compute_primed_integrals(
                    t_old, t_new, eqn, secondary, gridpoint_indices
                )
                I_1 += I_1_current
                I_2 += I_2_current
                I_3 += I_3_current
                
                # Add boundary contributions
                I_i[0] += self.compute_boundary_contribution(eqn, 'left', t_old, t_new)
                I_i[-1] += self.compute_boundary_contribution(eqn, 'right', t_old, t_new)
            
            # Get matrix coefficients for final solve
            a_, b_, c_ = self._assemble_matrix_coefficients(gridpoint_indices, eqn)
            
            # Combine all integral terms for the final right-hand side
            I_total = I_i + I_1 + I_2 + I_3
            
            # Compute RHS and solve system
            rhs = self._compute_rhs(y_current[eqn], I_total, a_, b_, c_)
            y_next = self.solve_tridiagonal(a_[1:], b_, c_[:-1], rhs)
            
            # Update solution array
            self.y[eqn, 1:-1, primary_interval + 1] = y_next
            
            if self.verbose:
                print(f"Completed equation {eqn + 1} out of {self.num_ODEs}...")
        
        # Set boundary conditions for next iteration
        current_time = t_start + self.hbar * num_secondaries
        for eqn in range(self.num_ODEs):
            self.set_boundary_values(
                self.y, 
                index=eqn, 
                time_index=primary_interval + 1, 
                current_time=current_time
            )

class Implicit_vectorized_fdm_simulation(Simulation):
    """ Vectorized implicit finite difference simulation.
    
    Implements implicit time stepping with vectorized operations for improved
    computational efficiency. Processes multiple spatial points simultaneously
    during integration while maintaining the same tridiagonal solving approach.
    
    Uses the same Thomas algorithm for tridiagonal systems but vectorizes the
    integration and primed integral computation steps for better performance
    on larger spatial grids.
    
    Attributes:
        Inherits all attributes from Simulation base class.
        
    Notes:
        Provides same functionality as non-vectorized version but with better
        performance through vectorized array operations during integration phases.
    """    
    def __init__(self, general_params, sympy_params):
        """ Initialize vectorized implicit finite difference simulation.
        
        Args:
            general_params (dict): Simulation parameters including grid, time stepping,
                boundary conditions, and solver options.
            sympy_params (dict): Symbolic equation definitions, auxiliary functions,
                spatial variables, and constants.
        """
        super().__init__(general_params, sympy_params)
        # Note: We'll remove the instance-based wrapper functions from the non-vectorized version
        # and instead create vectorized integrand functions on-demand
        
    def _initialize_auto_sympy(self, sympy_params):
        """ Initialize equation writer for vectorized implicit scheme.
        
        Uses same symbolic processing as non-vectorized version since vectorization
        occurs during numerical evaluation, not symbolic manipulation.
        
        Args:
            sympy_params (dict): Symbolic parameters including equations, auxiliaries,
                spatial variables, and constants.
        """
        self.evals = Implicit_fdm_equation_writer(
            sympy_params['list_equations'],
            sympy_params['list_auxiliary'],
            sympy_params['list_spatial_vars'],
            sympy_params['list_constants'],
            sympy_params['temporal_var'],
            self.r,
            self.index_range,
            self.factorials
        )
        self.evals.implicit_finite_diff_approx(
            z=sympy_params['list_spatial_vars'][0],     # spatial variable
            dz_sym=sympy_params['list_constants'][0][0] # grid spacing symbol
        )

    @staticmethod
    def solve_tridiagonal(a, b, c, d):
        """ Solve tridiagonal system using efficient Thomas algorithm.
        
        Reuses the optimized Thomas algorithm implementation from the non-vectorized
        version for consistent and efficient tridiagonal system solving.
        
        Args:
            a (np.ndarray): Lower diagonal elements (length n-1).
            b (np.ndarray): Main diagonal elements (length n).
            c (np.ndarray): Upper diagonal elements (length n-1).
            d (np.ndarray): Right-hand side vector (length n).
            
        Returns:
            np.ndarray: Solution vector x (length n).
        """
        n = len(d)
        
        # Declare buffer and output arrays
        c_prime = np.empty(n-1)
        d_prime = np.empty(n)
        x = np.empty(n)

        # Forward sweep - sequential
        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]
        
        for i in range(1, n-1):
            denominator = b[i] - a[i-1] * c_prime[i-1]
            c_prime[i] = c[i] / denominator
            d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / denominator
        
        # Last equation
        d_prime[n-1] = (d[n-1] - a[n-2] * d_prime[n-2]) / (b[n-1] - a[n-2] * c_prime[n-2])
        
        # Back substitution - sequential
        x[n-1] = d_prime[n-1]
        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]
            
        return x
    
    def get_lambda_inputs(self, gridpoint_indices):
        """ Construct inputs for lambda functions with consistent argument ordering.
        
        Reuses the input construction from non-vectorized version to maintain
        compatibility with equation writer lambda functions.
        
        Args:
            gridpoint_indices (np.ndarray): Array of interior grid point indices.
            
        Returns:
            list: Ordered arguments for lambda function evaluation.
        """
        total_size = 1 + len(self.evals.spatial_vars_list) + self.num_aux + self.num_ODEs * (1 + self.r)
        inputs = [None] * total_size
        
        idx = 0
        inputs[idx] = gridpoint_indices
        idx += 1
        
        # Spatial variables
        for var in range(len(self.evals.spatial_vars_list)):
            inputs[idx] = self.r_values
            idx += 1
            
        # Auxiliary values
        for aux in range(self.num_aux):
            inputs[idx] = self.aux[aux]
            idx += 1
            
        # Equations and derivatives
        for eqn in range(self.num_ODEs):
            inputs[idx] = self.U[eqn]
            idx += 1
            for tylr in range(self.r):
                inputs[idx] = self.f_array[eqn, tylr]
                idx += 1
                
        return inputs

    def _assemble_matrix_coefficients(self, gridpoint_indices, eqn):
        """ Assemble tridiagonal matrix coefficients using equation writer outputs.
        
        Reuses coefficient assembly from non-vectorized version since matrix
        coefficient computation is already efficient.
        
        Args:
            gridpoint_indices (np.ndarray): Interior point indices.
            eqn (int): Equation index for coefficient computation.
            
        Returns:
            tuple: (a_, b_, c_) arrays for tridiagonal system.
        """
        lambda_inputs = self.get_lambda_inputs(gridpoint_indices)
        
        eqn_str = str(eqn)
        a_key = 'a_' + eqn_str + '_lambdified'
        b_key = 'b_' + eqn_str + '_lambdified'  
        c_key = 'c_' + eqn_str + '_lambdified'
        
        c_dict = self.evals.c_dict
        
        a_ = c_dict[a_key](*lambda_inputs)
        b_ = c_dict[b_key](*lambda_inputs)
        c_ = c_dict[c_key](*lambda_inputs)
        
        return a_, b_, c_

    def _compute_rhs(self, y_current, I_i, a_, b_, c_):
        """ Compute right-hand side using vectorized operations.
        
        Reuses vectorized RHS computation from non-vectorized version.
        
        Args:
            y_current (np.ndarray): Current solution state at interior points.
            I_i (np.ndarray): Accumulated integration results.
            a_ (np.ndarray): Lower diagonal coefficients.
            b_ (np.ndarray): Main diagonal coefficients.
            c_ (np.ndarray): Upper diagonal coefficients.
            
        Returns:
            np.ndarray: Right-hand side vector for tridiagonal solve.
        """
        
        # Vectorized matrix multiplication
        rhs = b_ * y_current + I_i  # Diagonal terms + integration
        
        # Upper diagonal terms: c[:-1] * y[1:]
        rhs[:-1] += c_[:-1] * y_current[1:]
        
        # Lower diagonal terms: a[1:] * y[:-1]  
        rhs[1:] += a_[1:] * y_current[:-1]
        
        return rhs

    def compute_boundary_contribution(self, eqn, boundary_side, t_old, t_new):
        """ Compute boundary contributions for the vectorized scheme.
        
        Reuses boundary contribution computation from non-vectorized version
        since boundary handling logic is the same.
        
        Args:
            eqn (int): Equation index.
            boundary_side (str): 'left' or 'right' boundary.
            t_old (float): Start time of integration interval.
            t_new (float): End time of integration interval.
            
        Returns:
            float: Boundary contribution to the right-hand side.
        """
        side_index = 0 if boundary_side == 'left' else 1
        boundary_type = self.boundary_type[eqn][side_index]
        
        if boundary_type == 'fixed':
            return 0.0
        elif boundary_type == 'neumann':
            return 0.0  # To be implemented later
        elif boundary_type == 'open':
            return 0.0  # To be implemented later
        elif boundary_type == 'reflecting':
            return 0.0  # No additional contribution needed?
        elif boundary_type == 'time_dependent':
            return 0.0  # To be implemented later
        else:
            raise ValueError(f"Unknown boundary type: {boundary_type}")

    def _get_primed_boundary_values(self, eqn, t_old):
        """ Pre-compute boundary values for primed integral computation.
        
        Reuses boundary value computation from non-vectorized version.
        
        Args:
            eqn (int): Equation index.
            t_old (float): Time for boundary value evaluation.
            
        Returns:
            dict: Boundary values with 'left' and 'right' keys.
        """
        boundary_vals = {'left': (0.0, 0.0), 'right': (0.0, 0.0)}
        
        # Left boundary
        if self.boundary_type[eqn][0] == 'fixed':
            boundary_vals['left'] = (self.boundary_vals[eqn][0], 0.0)
        elif self.boundary_type[eqn][0] == 'reflecting':
            boundary_vals['left'] = (self.U[eqn, 1], 0.0)  # Use adjacent interior point
        elif self.boundary_type[eqn][0] == 'time_dependent':
            alpha_left = self.time_dependent_funcs[eqn][0](t_old, self.r_values[0])
            boundary_vals['left'] = (alpha_left, 0.0)
        else:
            boundary_vals['left'] = (self.U[eqn, 0], 0.0)
        
        # Right boundary  
        if self.boundary_type[eqn][1] == 'fixed':
            boundary_vals['right'] = (self.boundary_vals[eqn][1], 0.0)
        elif self.boundary_type[eqn][1] == 'reflecting':
            boundary_vals['right'] = (self.U[eqn, -2], 0.0)  # Use adjacent interior point
        elif self.boundary_type[eqn][1] == 'time_dependent':
            alpha_right = self.time_dependent_funcs[eqn][1](t_old, self.r_values[-1])
            boundary_vals['right'] = (alpha_right, 0.0)
        else:
            boundary_vals['right'] = (self.U[eqn, -1], 0.0)
            
        return boundary_vals

    def create_vectorized_integrand_func(self, lambda_func, coeffs_array, boundary_values=None):
        """ Create vectorized integrand function for primed integrals.
        
        Wraps primed lambda functions to handle vectorized evaluation across
        multiple spatial points and integration samples simultaneously.
        
        Args:
            lambda_func (callable): The primed lambda function (a', b', or c').
            coeffs_array (np.ndarray): Coefficient array of shape (N_points, 3*(r+2)).
            boundary_values (dict, optional): Alpha/alpha_dot values. Defaults to None.
            
        Returns:
            callable: Vectorized integrand function handling multiple spatial points.
            
        Notes:
            Handles both 1D and 2D time arrays from quad integration.
            Evaluates lambda function for each spatial point and sample combination.
            Sets boundary values to zero if not provided (for interior points).
        """
        def vectorized_integrand(t, t0, C_dummy):
            """
            Vectorized integrand function that processes multiple spatial points.
            
            Args:
                t: Time values - shape (N_points, n_samples) from quad integration
                t0: Initial time values - shape (N_points, 1)
                C_dummy: Dummy coefficient array (not used, but required by interface)
                
            Returns:
                Array of integrand values - shape (N_points, n_samples)
            """
            # Handle the shape from quad integration: t is (N_points, n_samples), t0 is (N_points, 1)
            N_points = coeffs_array.shape[0]
            
            # Set default boundary values if not provided
            if boundary_values is None:
                alpha_left = alpha_right = 0.0
                alpha_dot_left = alpha_dot_right = 0.0
            else:
                alpha_left = boundary_values.get('alpha_left', 0.0)
                alpha_right = boundary_values.get('alpha_right', 0.0)
                alpha_dot_left = boundary_values.get('alpha_dot_left', 0.0)
                alpha_dot_right = boundary_values.get('alpha_dot_right', 0.0)
            
            # Determine output shape based on t
            if t.ndim == 1:
                # t is 1D, so we have N_points values
                n_samples = 1
                t_reshaped = t.reshape(-1, 1)
            else:
                # t is 2D (N_points, n_samples)
                n_samples = t.shape[1]
                t_reshaped = t
            
            # Initialize result array
            results = np.zeros((N_points, n_samples))
            
            # Evaluate lambda function for each spatial point and each sample
            for i in range(N_points):
                for j in range(n_samples):
                    results[i, j] = lambda_func(
                        t_reshaped[i, j], t0[i, 0],  # Use t0[i, 0] to get scalar
                        *coeffs_array[i],  # Unpack coefficients for this point
                        alpha_left, alpha_right, alpha_dot_left, alpha_dot_right
                    )
            
            # Return appropriate shape
            if n_samples == 1:
                return results.flatten()
            else:
                return results
        
        return vectorized_integrand

    def _integrate_secondary_vectorized(self, t_old, t_new, eqn, secondary, gridpoint_indices):
        """ Vectorized secondary interval integration.
        
        Performs integration over a secondary interval using vectorized operations
        to process all interior spatial points simultaneously.
        
        Args:
            t_old (float): Start time of secondary interval.
            t_new (float): End time of secondary interval.
            eqn (int): Index of current equation.
            secondary (int): Index of current secondary interval.
            gridpoint_indices (np.ndarray): Array of interior point indices.
            
        Returns:
            np.ndarray: Array of integration results for interior points.
            
        Notes:
            Prepares vectorized input arrays and calls vectorized integration.
            Significantly faster than non-vectorized version for large grids.
        """
        N_interior = len(gridpoint_indices)
        
        # Prepare vectorized inputs
        tlims_array = np.tile([t_old, t_new], (N_interior, 1))  # Shape: (N_interior, 2)
        C_array = self.ll[eqn, 1:, :, secondary].T  # Shape: (N_interior, r+1)
        t0_array = np.full(N_interior, t_old)  # Shape: (N_interior,)
        
        # Single vectorized integration call
        results = kacewicz_functions.integrate(
            integrand_function=self.integrand_func,
            tlims=tlims_array,
            n_samples=self.n_samples,
            integral_mode=self.integral_mode,
            C=C_array,
            t0=t0_array,
            quantum=self.quantum,
            M=self.M,
            delta=self.delta,
            use_vectorized=True
        )
        
        return results

    def _compute_primed_integrals_vectorized(self, t_old, t_new, eqn, secondary, gridpoint_indices):
        """ Vectorized computation of primed integrals with non-vectorized boundaries.
        
        Vectorizes middle point computation while keeping boundary points non-vectorized
        due to boundary condition complexity. Provides performance improvement for
        interior points while maintaining accuracy at boundaries.
        
        Args:
            t_old (float): Start time of secondary interval.
            t_new (float): End time of secondary interval.
            eqn (int): Index of current equation.
            secondary (int): Index of current secondary interval.
            gridpoint_indices (np.ndarray): Array of interior point indices.
            
        Returns:
            tuple: (I_1, I_2, I_3) arrays of primed integral contributions.
            
        Notes:
            Creates vectorized integrand functions for middle points (indices 1 to N-3).
            Handles boundary points (indices 0 and N-1) with individual integration
            calls due to boundary condition requirements.
        """
        N_interior = len(gridpoint_indices)
        
        # Pre-allocate results
        I_1 = np.zeros(N_interior)
        I_2 = np.zeros(N_interior) 
        I_3 = np.zeros(N_interior)
        
        # Cache dictionary lookups and get lambda functions
        eqn_str = str(eqn)
        c_dict = self.evals.c_dict
        
        # Get coefficient array once
        C_all = self.ll[:, :, :, secondary]
        bL, bLd, bR, bRd = _boundary_arrays(self, t_old)
        zeros = np.zeros(self.num_ODEs)
        
        # === VECTORIZED MIDDLE POINTS ===
        if N_interior > 2:  # Only if we have middle points
            middle_indices = np.arange(1, N_interior - 1)
            n_middle = len(middle_indices)
            
            if n_middle > 0:
                # Prepare vectorized coefficient arrays for middle points
                middle_coeffs = np.zeros((n_middle, 3 * self.num_ODEs * (self.r + 2)))
                
                for idx, i in enumerate(middle_indices):
                    middle_coeffs[idx] = _point_coeffs(self, C_all, i, N_interior)
                
                # Get middle lambda functions
                a_lambda = c_dict[f'a_{eqn_str}_prime_integrand_lambda_middle']
                b_lambda = c_dict[f'b_{eqn_str}_prime_integrand_lambda_middle']
                c_lambda = c_dict[f'c_{eqn_str}_prime_integrand_lambda_middle']
                
                # Create vectorized integrand functions (no boundary values for middle points)
                a_integrand = self.create_vectorized_integrand_func(a_lambda, middle_coeffs)
                b_integrand = self.create_vectorized_integrand_func(b_lambda, middle_coeffs)
                c_integrand = self.create_vectorized_integrand_func(c_lambda, middle_coeffs)
                
                # Prepare vectorized integration inputs
                tlims_array = np.tile([t_old, t_new], (n_middle, 1))
                t0_array = np.full(n_middle, t_old)
                dummy_C = np.ones((n_middle, 1))  # Dummy coefficients array
                
                # Vectorized integrations for middle points
                I_1_middle = kacewicz_functions.integrate(
                    integrand_function=a_integrand,
                    tlims=tlims_array,
                    n_samples=self.n_samples,
                    integral_mode=self.integral_mode,
                    C=dummy_C,
                    t0=t0_array,
                    quantum=self.quantum,
                    M=self.M,
                    delta=self.delta,
                    use_vectorized=True
                )
                
                I_2_middle = kacewicz_functions.integrate(
                    integrand_function=b_integrand,
                    tlims=tlims_array,
                    n_samples=self.n_samples,
                    integral_mode=self.integral_mode,
                    C=dummy_C,
                    t0=t0_array,
                    quantum=self.quantum,
                    M=self.M,
                    delta=self.delta,
                    use_vectorized=True
                )
                
                I_3_middle = kacewicz_functions.integrate(
                    integrand_function=c_integrand,
                    tlims=tlims_array,
                    n_samples=self.n_samples,
                    integral_mode=self.integral_mode,
                    C=dummy_C,
                    t0=t0_array,
                    quantum=self.quantum,
                    M=self.M,
                    delta=self.delta,
                    use_vectorized=True
                )
                
                # Store middle results
                I_1[1:-1] = I_1_middle
                I_2[1:-1] = I_2_middle
                I_3[1:-1] = I_3_middle
        
        # === NON-VECTORIZED BOUNDARY POINTS ===
        # Left boundary point (i=0)
        if N_interior > 0:
            a_lambda = c_dict[f'a_{eqn_str}_prime_integrand_lambda_left']
            b_lambda = c_dict[f'b_{eqn_str}_prime_integrand_lambda_left']
            c_lambda = c_dict[f'c_{eqn_str}_prime_integrand_lambda_left']
            
            alpha_left, alpha_dot_left = bL, bLd
            
            # Coefficient handling for left boundary
            all_coeffs = _point_coeffs(self, C_all, 0, N_interior)
            
            # Single point integrations for left boundary
            I_1[0] = a_lambda(t_old, t_old, *all_coeffs, alpha_left, zeros, alpha_dot_left, zeros)
            I_2[0] = b_lambda(t_old, t_old, *all_coeffs, alpha_left, zeros, alpha_dot_left, zeros)
            I_3[0] = c_lambda(t_old, t_old, *all_coeffs, alpha_left, zeros, alpha_dot_left, zeros)
            
            # Actually integrate (not just evaluate at t_old)
            dummy_C_single = np.ones(1)
            I_1[0] = kacewicz_functions.integrate(
                integrand_function=lambda t, t0, C: a_lambda(t, t0, *all_coeffs, alpha_left, zeros, alpha_dot_left, zeros),
                tlims=np.array([t_old, t_new]),
                n_samples=self.n_samples,
                integral_mode=self.integral_mode,
                C=dummy_C_single,
                t0=t_old,
                quantum=self.quantum,
                M=self.M,
                delta=self.delta,
                use_vectorized=False
            )
            
            I_2[0] = kacewicz_functions.integrate(
                integrand_function=lambda t, t0, C: b_lambda(t, t0, *all_coeffs, alpha_left, zeros, alpha_dot_left, zeros),
                tlims=np.array([t_old, t_new]),
                n_samples=self.n_samples,
                integral_mode=self.integral_mode,
                C=dummy_C_single,
                t0=t_old,
                quantum=self.quantum,
                M=self.M,
                delta=self.delta,
                use_vectorized=False
            )
            
            I_3[0] = kacewicz_functions.integrate(
                integrand_function=lambda t, t0, C: c_lambda(t, t0, *all_coeffs, alpha_left, zeros, alpha_dot_left, zeros),
                tlims=np.array([t_old, t_new]),
                n_samples=self.n_samples,
                integral_mode=self.integral_mode,
                C=dummy_C_single,
                t0=t_old,
                quantum=self.quantum,
                M=self.M,
                delta=self.delta,
                use_vectorized=False
            )
        
        # Right boundary point (i=N_interior-1)
        if N_interior > 1:
            a_lambda = c_dict[f'a_{eqn_str}_prime_integrand_lambda_right']
            b_lambda = c_dict[f'b_{eqn_str}_prime_integrand_lambda_right']
            c_lambda = c_dict[f'c_{eqn_str}_prime_integrand_lambda_right']
            
            alpha_right, alpha_dot_right = bR, bRd
            
            # Coefficient handling for right boundary
            i = N_interior - 1
            all_coeffs = _point_coeffs(self, C_all, i, N_interior)
            
            # Single point integrations for right boundary
            dummy_C_single = np.ones(1)
            I_1[-1] = kacewicz_functions.integrate(
                integrand_function=lambda t, t0, C: a_lambda(t, t0, *all_coeffs, zeros, alpha_right, zeros, alpha_dot_right),
                tlims=np.array([t_old, t_new]),
                n_samples=self.n_samples,
                integral_mode=self.integral_mode,
                C=dummy_C_single,
                t0=t_old,
                quantum=self.quantum,
                M=self.M,
                delta=self.delta,
                use_vectorized=False
            )
            
            I_2[-1] = kacewicz_functions.integrate(
                integrand_function=lambda t, t0, C: b_lambda(t, t0, *all_coeffs, zeros, alpha_right, zeros, alpha_dot_right),
                tlims=np.array([t_old, t_new]),
                n_samples=self.n_samples,
                integral_mode=self.integral_mode,
                C=dummy_C_single,
                t0=t_old,
                quantum=self.quantum,
                M=self.M,
                delta=self.delta,
                use_vectorized=False
            )
            
            I_3[-1] = kacewicz_functions.integrate(
                integrand_function=lambda t, t0, C: c_lambda(t, t0, *all_coeffs, zeros, alpha_right, zeros, alpha_dot_right),
                tlims=np.array([t_old, t_new]),
                n_samples=self.n_samples,
                integral_mode=self.integral_mode,
                C=dummy_C_single,
                t0=t_old,
                quantum=self.quantum,
                M=self.M,
                delta=self.delta,
                use_vectorized=False
            )

        return I_1, I_2, I_3

    def _handle_custom_output_vectorized(self, t_old, eqn, secondary, y_current, custom_time, gridpoint_indices):
        """ Vectorized custom output computation.
        
        Handles custom output times using vectorized integration for improved
        performance while maintaining the same accuracy as non-vectorized version.
        
        Args:
            t_old (float): Start time of secondary interval.
            eqn (int): Index of current equation.
            secondary (int): Index of current secondary interval.
            y_current (np.ndarray): Current solution state for this equation.
            custom_time (float): Requested output time.
            gridpoint_indices (np.ndarray): Array of interior point indices.
            
        Notes:
            Uses vectorized integration and primed integral computation.
            Updates custom output arrays and HDF5 file when complete.
        """
        if custom_time == t_old:
            # At start of interval, use current solution
            self.custom_outputs[eqn, 1:-1, self.t_counter] = y_current
        else:
            # Use vectorized integration up to custom time
            custom_integration = self._integrate_secondary_vectorized(
                t_old, custom_time, eqn, secondary, gridpoint_indices
            )
            
            # Compute vectorized primed integrals up to custom time
            I_1, I_2, I_3 = self._compute_primed_integrals_vectorized(
                t_old, custom_time, eqn, secondary, gridpoint_indices
            )
            
            # Combine all integral contributions
            total_integration = custom_integration + I_1 + I_2 + I_3
            
            # Get matrix coefficients and solve system
            a_, b_, c_ = self._assemble_matrix_coefficients(gridpoint_indices, eqn)
            rhs = self._compute_rhs(y_current, total_integration, a_, b_, c_)
            custom_solution = self.solve_tridiagonal(a_[1:], b_, c_[:-1], rhs)
            
            self.custom_outputs[eqn, 1:-1, self.t_counter] = custom_solution
        
        # If this is the last equation, handle boundaries and update counter
        if eqn == self.num_ODEs - 1:
            for i in range(self.num_ODEs):
                self.set_boundary_values(
                    self.custom_outputs, 
                    i, 
                    time_index=self.t_counter, 
                    current_time=custom_time
                )
            self.update_save_custom_h5(self.t_counter)
            self.t_counter += 1

    def IntegrateGij(self, t_start, primary_interval):
        """ Vectorized implicit integration scheme.
        
        Performs integration over all secondary intervals within a primary interval
        using vectorized operations for improved computational efficiency while
        maintaining the same algorithmic structure as the non-vectorized version.
        
        Args:
            t_start (float): Starting time of the current primary interval.
            primary_interval (int): Index of the current primary interval.
            
        Notes:
            Uses vectorized secondary integration and primed integral computation.
            Reuses non-vectorized components (matrix assembly, boundary contributions)
            where vectorization provides minimal benefit or adds complexity.
        """
        # Initialize arrays
        N_interior = self.N - 2
        gridpoint_indices = np.arange(1, self.N-1)
        
        # Current solution state
        y_current = self.y[:, 1:-1, primary_interval].copy()
        
        # Pre-compute time step and secondary count
        num_secondaries = self.n**(self.k-1)
        
        # Loop over each equation in the system
        for eqn in range(self.num_ODEs):
            # Pre-allocate accumulator arrays for all integral terms
            I_i = np.zeros(N_interior)
            I_1 = np.zeros(N_interior)
            I_2 = np.zeros(N_interior)
            I_3 = np.zeros(N_interior)
            
            # Accumulate integrations over all secondaries in this primary interval
            for secondary in range(num_secondaries):
                t_old = t_start + (secondary * self.hbar)
                t_new = t_start + ((secondary + 1) * self.hbar)
                
                # Handle custom output times if needed
                if self.t_counter < self.t_output_length:
                    custom_time = self.t_outputs[self.t_counter]
                    
                    # Add tolerance for floating point comparison
                    tolerance = 1e-10 * max(abs(custom_time), abs(t_new))
                    
                    if t_old <= custom_time < t_new or abs(custom_time - t_new) < tolerance:
                        self._handle_custom_output_vectorized(
                            t_old, eqn, secondary,
                            y_current[eqn], custom_time, gridpoint_indices
                        )
                
                # Vectorized integration over this secondary interval for main right-hand side
                I_i += self._integrate_secondary_vectorized(
                    t_old, t_new, eqn, secondary, gridpoint_indices
                )
                
                # Vectorized computation of primed integrals
                I_1_current, I_2_current, I_3_current = self._compute_primed_integrals_vectorized(
                    t_old, t_new, eqn, secondary, gridpoint_indices
                )
                I_1 += I_1_current
                I_2 += I_2_current
                I_3 += I_3_current
                
                # Add boundary contributions (reuse from non-vectorized)
                I_i[0] += self.compute_boundary_contribution(eqn, 'left', t_old, t_new)
                I_i[-1] += self.compute_boundary_contribution(eqn, 'right', t_old, t_new)
            
            # Get matrix coefficients for final solve
            a_, b_, c_ = self._assemble_matrix_coefficients(gridpoint_indices, eqn)
            
            # Combine all integral terms for the final right-hand side
            I_total = I_i + I_1 + I_2 + I_3
            
            # Compute RHS and solve system
            rhs = self._compute_rhs(y_current[eqn], I_total, a_, b_, c_)
            y_next = self.solve_tridiagonal(a_[1:], b_, c_[:-1], rhs)
            
            # Update solution array
            self.y[eqn, 1:-1, primary_interval + 1] = y_next
            
            if self.verbose:
                print(f"Completed equation {eqn + 1} out of {self.num_ODEs}...")
        
        # Set boundary conditions for next iteration
        current_time = t_start + self.hbar * num_secondaries
        for eqn in range(self.num_ODEs):
            self.set_boundary_values(
                self.y, 
                index=eqn, 
                time_index=primary_interval + 1, 
                current_time=current_time
            )