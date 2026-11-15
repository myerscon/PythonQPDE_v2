"""
Main script for the Python Quantum Partial Differential Equation (QPDE) solver.

This script contains the Simulation class, which handles the setup and execution
of PDE simulations. The simulation parameters and equations are defined in a separate
input file, which is imported when creating and running the Simulation class.

The code utilizes SymPy to analytically derive expressions for the driver functions,
derivatives of driver functions, and auxiliary functions. This functionality is
contained in the auto_sympy.py script, which itself uses functions from custom_sympy.py.

The integration subroutines were adapted from code by Frank Gaitan and are stored in
kacewicz_functions.py.

Required Python packages:
- NumPy
- SciPy
- SymPy
- h5py

To run the code:
1. Install the necessary packages in your environment.
2. Execute the script using:
   $ python run.py input.py
   Replace 'input.py' with the name of the input deck you want to run.

The outputs will be saved in an output directory within the directory where main.py
is located. The output directory name is specified in the input script.

Author: Conner Myers
Created: 11/14/23
Last modified: [Insert date of last modification]
"""

import os # Only used for creating the output directory if it does not already exist
import numpy as np
import h5py as h5
import csv
import math
import time
import kacewicz_functions
from abc import ABC, abstractmethod

class Simulation(ABC):
    """
    Abstract base class for PDE simulation using quantum or classical integration methods.

    This class provides the foundational framework for solving partial differential equations
    using the quantum PDE (QPDE) solver methodology. It handles grid setup, time stepping,
    boundary conditions, file I/O, and the overall simulation workflow.

    The simulation uses a hierarchical time integration scheme with primary and secondary
    intervals, supporting both explicit and implicit finite difference methods.

    Attributes:
        N (int): Number of spatial grid points.
        xlims (tuple): Spatial domain limits (left, right).
        xs (np.ndarray): Spatial grid coordinates.
        r_values (np.ndarray): Alias for xs, used for 'r' coordinate references.
        h (float): Spatial grid spacing.
        r (int): Maximum Taylor series derivative order.
        num_ODEs (int): Number of coupled ODEs in the system.
        num_aux (int): Number of auxiliary functions.
        hbar (float): Time step size.
        n (int): Number of primary time intervals.
        k (int): Hierarchical time stepping parameter.
        U (np.ndarray): Current solution state array, shape (num_ODEs, N).
        y (np.ndarray): Solution at primary intervals, shape (num_ODEs, N, n+1).
        quantum (bool): Whether to use quantum integration algorithms.
        
    Notes:
        This is an abstract base class. Concrete implementations must define:
        - _initialize_auto_sympy(): Set up symbolic equation processing
        - IntegrateGij(): Perform time integration over primary intervals
    """
    def __init__(self, general_params, sympy_params) -> None:
        """
        Initialize the simulation with parameters and set up arrays.

        Sets up the spatial grid, time stepping parameters, solution arrays, initial
        and boundary conditions, and prepares the symbolic equation processing system.

        Args:
            general_params (dict): Dictionary containing numerical and physical parameters:
                - N (int): Number of spatial grid points
                - xlims (tuple): Spatial domain bounds
                - taylor_r (int): Taylor series order
                - dt (float): Time step size
                - num_ODEs (int): Number of equations
                - initial_cond_type (list): Initial condition types per equation
                - initial_cond_vals (list): Initial condition values
                - boundary_type (list): Boundary condition types
                - quantum (bool): Enable quantum integration
                - epsilon (float): Integration error tolerance
                - And other simulation parameters
            sympy_params (dict): Dictionary containing symbolic parameters:
                - list_equations: Symbolic equation definitions
                - list_auxiliary: Auxiliary function definitions
                - list_spatial_vars: Spatial variables
                - list_constants: Physical constants
                - temporal_var: Time variable

        Returns:
            None

        Raises:
            KeyError: If required parameters are missing from input dictionaries.
        """
        # Grid parameters
        self.N = general_params['N']
        self.xlims = general_params['xlims']
        self.xs = np.linspace(self.xlims[0], self.xlims[1], self.N, endpoint=True)
        self.r_values = self.xs  # Store the 'r' values # 10/9 addition
        self.h = (self.xlims[1] - self.xlims[0]) / (self.N - 1)

        # Simulation parameters
        self.r = general_params['taylor_r']
        self.num_ODEs = general_params['num_ODEs']
        self.num_aux = general_params['num_aux']
        self.use_vectorized = general_params['use_vectorized']
        self.quantum = general_params['quantum']
        self.sim_type = 'quantum' if self.quantum else 'classical'
        self.integral_mode = general_params['integral_mode']
        self.index_range = general_params['index_range']
        self.directory_name = general_params['directory_name']
        self.verbose = general_params['verbose']
        self.time_discretization = general_params.get('time_discretization', 'explicit')  # Default to explicit scheme

        # Time parameters
        self.hbar = general_params['dt']
        self.t_outputs = general_params['time_outputs']
        self.t_output_length = len(self.t_outputs)
        self.t_counter = 0
        self.current_primary = 0
        self.current_secondary = 0
        self.secondary_save_frequency = general_params.get('secondary_save_frequency', 0)  # Added Nov 19

        # Integration parameters
        self.epsilon = general_params['epsilon']
        if (self.time_discretization=='explicit'):
            self.N_tot = general_params['N_tot']
            self.n, self.k = kacewicz_functions.calculate_partition_parameters(self.time_discretization, self.epsilon, [self.N_tot])
            print("EXPLICIT SCHEME")
            print("epsilon: "+str(self.epsilon))
            print("N_tot: "+str(self.N_tot))
        elif (self.time_discretization=='implicit'):
            self.total_time = general_params['total_time']
            self.n, self.k = kacewicz_functions.calculate_partition_parameters(self.time_discretization, self.epsilon, [self.total_time,self.hbar])
            print("IMPLICIT SCHEME")
            print("epsilon: "+str(self.epsilon))
            print("total_time: "+str(self.total_time))
            print("hbar: "+str(self.hbar))
        elif (self.time_discretization=='implicit_pick_nk'):
            print("IMPLICIT SCHEME - FORCE PICK N and K")
            # n (primaries) and k (levels) are deck inputs; defaults are the values used so far
            self.n = general_params.get('n', 126)
            self.k = general_params.get('k', 3)
            # Pick hbar based on User supplied T
            self.total_time = general_params['total_time']
            self.hbar = (self.total_time) / (self.n**self.k)
            print("epsilon: "+str(self.epsilon))
            print("total_time: "+str(self.total_time))
            print("hbar: "+str(self.hbar))
        print("primaries: " + str(self.n))
        print("k: " + str(self.k))
        print("secondaries per primary: " + str(self.n**(self.k-1)))
        # These are used inside IntegrateGij child methods:
        self.n_samples = general_params['n_samples']
        self.M = self.n**(self.k-1)
        self.delta = general_params['delta']

        # Initialize arrays
        self.U = np.zeros((self.num_ODEs, self.N))
        self.aux = np.zeros((self.num_aux, self.N))
        self.f_array = np.zeros((self.num_ODEs, self.r+1, self.N))
        self.custom_outputs = np.zeros((self.num_ODEs, self.N, self.t_output_length))
        print("self.num_ODEs: "+str(self.num_ODEs))
        print("self.N: "+str(self.N))
        print("self.n+1: "+str(self.n+1))
        self.y = np.zeros((self.num_ODEs, self.N, self.n+1))
        self.mat = np.zeros((self.num_ODEs, self.r+2, self.N-2))
        self.ll = np.zeros((self.num_ODEs, self.r+2, self.N-2, self.n**(self.k-1)))

        # For storing alpha values at each secondary
        self.alpha = np.zeros((self.num_ODEs, self.N, self.n**(self.k-1)))
        # For storing alpha time derivatives at each secondary
        self.alpha_dot = np.zeros((self.num_ODEs, self.N, self.n**(self.k-1)))

        # Runtime tracking attributes
        self.simulation_start_time = None
        self.primary_runtimes = []  # Store runtime for each primary interval

        # Set initial conditions
        self._set_initial_conditions(general_params)

        # Set boundary conditions
        self.time_dependent_funcs = general_params.get('time_dependent_funcs', None)
        self.boundary_type = np.array(general_params['boundary_type'])
        self.boundary_vals = np.array(general_params['boundary_vals'])
        self._set_boundary_conditions()

        # Pre-compute factorial values
        self.factorials = np.array([math.factorial(i) for i in range(self.r + 2)])

        # Implicit coefficients must use the real secondary step: overwrite the deck's numeric hbar
        if self.time_discretization in ('implicit', 'implicit_pick_nk'):
            symbols, values = sympy_params['list_constants']
            values = list(values)
            for idx, sym in enumerate(symbols):
                if sym.name == r'\bar{h}':
                    values[idx] = self.hbar
            sympy_params = {**sympy_params, 'list_constants': (symbols, values)}

        # Initialize auto_sympy
        self._initialize_auto_sympy(sympy_params)

        # Create integrand functions
        self.integrand_func = self._create_integrand_func()

    def _set_initial_conditions(self, general_params):
        """
        Set initial conditions for all equations in the system.

        Applies initial conditions based on the specified type for each equation.
        Supports 'fixed' (constant value) and 'specified' (arbitrary array) types.

        Args:
            general_params (dict): Contains 'initial_cond_type' and 'initial_cond_vals'
                where each is a list with one entry per equation.

        Returns:
            None

        Notes:
            - 'fixed': Sets uniform value across all spatial points
            - 'specified': Uses provided array of values at each grid point
            - Updates both self.U and self.y[:,:,0] arrays
        """
        for i in range(self.num_ODEs):
            if general_params['initial_cond_type'][i] == 'fixed':
                self.U[i] = general_params['initial_cond_vals'][i] * np.ones(self.N)
            elif general_params['initial_cond_type'][i] == 'specified':
                self.U[i] = general_params['initial_cond_vals'][i]
        self.y[:,:,0] = self.U

    def _set_boundary_conditions(self):
        """
        Apply boundary conditions to the initial state.

        Sets boundary values for both the current solution array (U) and the
        primary interval solution array (y) at time index 0.

        Returns:
            None

        Notes:
            Called during initialization to ensure consistent boundary conditions
            across all solution arrays from the start.
        """
        initial_time = 0.0  # Set the initial time to 0
        for i in range(self.num_ODEs):
            self.set_boundary_values(self.U, index=i, current_time=initial_time)
            self.set_boundary_values(self.y, index=i, time_index=0, current_time=initial_time)

    @abstractmethod
    def _initialize_auto_sympy(self, sympy_params):
        """
        Initialize symbolic equation processing system.

        Abstract method that must be implemented by concrete simulation classes.
        Sets up the equation writer and performs finite difference approximations
        appropriate for the specific solution method (explicit/implicit).

        Args:
            sympy_params (dict): Symbolic parameters including equations, variables,
                constants, and temporal variable definitions.

        Returns:
            None

        Notes:
            Concrete implementations typically:
            1. Create an equation writer instance
            2. Call finite difference approximation methods
            3. Store the resulting lambda functions for evaluation
        """
        pass

    def _create_integrand_func(self):
        """
        Construct the integrand function for time integration.

        Creates a function that evaluates Taylor polynomial expansions of the
        driver functions over specified time intervals. Supports both vectorized
        and non-vectorized operations.

        Returns:
            function: Integrand function with signature:
                integrand_func(t, t0, C) -> np.ndarray
                where:
                - t: Time points for evaluation (1D array)
                - t0: Initial time (scalar or array)
                - C: Taylor polynomial coefficients
                
        Notes:
            The integrand implements:
            f(t) = Σ(i=0 to r) C[i] * (t-t0)^i / i!
            
            Handles both single-point and vectorized coefficient arrays automatically.
        """
        def integrand_func(t, t0, C):
            t_diff = t - t0
            if C.ndim == 1:  # Non-vectorized case
                return sum(C[i] * (t_diff ** i) / self.factorials[i] for i in range(self.r + 1))
            else:  # Vectorized case
                powers = np.arange(self.r + 1)
                terms = C[:, :, np.newaxis] * (t_diff[:, np.newaxis, :] ** powers[np.newaxis, :, np.newaxis]) / self.factorials[np.newaxis, :(self.r + 1), np.newaxis]
                return np.sum(terms, axis=1)
        return integrand_func

    def initial_save_h5(self):
        """
        Create output directory and initialize HDF5 output file.

        Sets up the complete HDF5 file structure including datasets for primary
        solutions, custom output times, spatial grid, and optional secondary
        interval storage.

        Returns:
            None

        Side Effects:
            - Creates output directory if it doesn't exist
            - Creates HDF5 file with initial datasets and metadata
            - Sets up secondary storage if secondary_save_frequency > 0

        Notes:
            The HDF5 file structure includes:
            - 'fields': Primary interval solutions
            - 'custom_fields': Solutions at user-specified times
            - 'spatial_grid': Spatial coordinate array
            - 'primary_interval_times': Time values for primary intervals
            - 'custom_output_times': User-specified output times
            - Optional: 'secondary_fields' and 'secondary_times'
        """
        if not os.path.exists(self.directory_name):
            os.makedirs(self.directory_name)
        
        # Calculate primary interval times
        times = np.zeros(self.n+1)
        for i in range(self.n+1):
            times[i] = i*(self.n**(self.k-1))*self.hbar
        
        # Create h5py storage arrays
        with h5.File(self.directory_name+'output.hdf5','w') as f:
            # Create standard datasets
            f.create_dataset('fields', data=self.y)
            f.create_dataset('custom_fields', data=self.custom_outputs)
            f.create_dataset('spatial_grid', data=self.xs)
            f.create_dataset('primary_interval_times', data=times)
            f.create_dataset('custom_output_times', data=self.t_outputs)
            
            # Only create secondary datasets if save frequency > 0
            if self.secondary_save_frequency > 0:
                num_secondaries = self.n**(self.k-1)
                num_saved_secondaries = num_secondaries // self.secondary_save_frequency + (1 if num_secondaries % self.secondary_save_frequency else 0)
                secondary_times = np.zeros((self.n, num_saved_secondaries))
                
                for i in range(self.n):
                    for j in range(num_saved_secondaries):
                        secondary_idx = j * self.secondary_save_frequency
                        if secondary_idx < num_secondaries:
                            secondary_times[i,j] = (i * num_secondaries + secondary_idx) * self.hbar
                
                f.create_dataset('secondary_fields', shape=(self.num_ODEs, self.N, self.n, num_saved_secondaries))
                f.create_dataset('secondary_times', data=secondary_times)
            
            # Create attributes
            f.attrs['current_primary'] = 0
            f.attrs['current_custom_output'] = 0
            f.attrs['complete_flag'] = False
            f.attrs['sim_type'] = self.sim_type
            f.attrs['save_secondaries'] = bool(self.secondary_save_frequency > 0)

    def update_save_secondary_h5(self, primary, secondary):
        """
        Save field values at specified secondary intervals.

        Conditionally saves secondary interval solutions based on the configured
        save frequency. Only saves if secondary_save_frequency > 0 and the current
        secondary index is a multiple of the save frequency.

        Args:
            primary (int): Index of current primary interval.
            secondary (int): Index of current secondary interval within the primary.

        Returns:
            None

        Notes:
            - Skips saving if secondary_save_frequency <= 0
            - Saves to 'secondary_fields' dataset in HDF5 file
            - Used for detailed temporal resolution analysis
        """
        # Skip if secondary saves are disabled
        if self.secondary_save_frequency <= 0:
            return
            
        if secondary % self.secondary_save_frequency == 0:
            saved_idx = secondary // self.secondary_save_frequency
            with h5.File(self.directory_name + 'output.hdf5', 'r+') as f:
                secondary_fields = f['secondary_fields']
                secondary_fields[:,:,primary,saved_idx] = self.U

    def update_save_h5(self, primary):
        """
        Update HDF5 file with current primary interval solution.

        Saves the solution state at the completion of a primary interval and
        updates simulation metadata attributes.

        Args:
            primary (int): Index of the completed primary interval.

        Returns:
            None

        Side Effects:
            - Updates 'fields' dataset with current and next states
            - Updates 'current_primary' attribute
            - Sets 'complete_flag' to True if this is the final interval

        Notes:
            Saves both the current state and the computed next state to ensure
            complete solution coverage across all primary intervals.
        """
        with h5.File(self.directory_name + 'output.hdf5', 'r+') as f:
            # Update the dataset for the current primary interval
            fields = f['fields']
            
            # Save both the current and next state (if available)
            fields[:, :, primary] = self.y[:, :, primary]
            
            # If we've calculated the next state, save it too
            if primary < self.n:  # Make sure we don't go out of bounds
                fields[:, :, primary + 1] = self.y[:, :, primary + 1]
            
            # Update the stored simulation attributes
            f.attrs['current_primary'] = primary + 1
            if ((primary + 1) == self.n):
                f.attrs['complete_flag'] = True

    def update_save_custom_h5(self, counter):
        """
        Update HDF5 file with solution at custom output time.

        Saves simulation state at user-specified custom output times and updates
        the corresponding metadata.

        Args:
            counter (int): Index of the current custom output time.

        Returns:
            None

        Side Effects:
            - Updates 'custom_fields' dataset
            - Updates 'current_custom_output' attribute

        Notes:
            Custom output times allow users to get solutions at specific times
            of interest that may not align with primary interval boundaries.
        """
        with h5.File(self.directory_name + 'output.hdf5', 'r+') as f:
            # Update the dataset for the current custom output counter
            custom_fields = f['custom_fields']
            custom_fields[:,:,counter] = self.custom_outputs[:,:,counter] # currently looping over whole array
            # Update the stored simulation attributes
            f.attrs['current_custom_output'] = counter

    def save_primary_snapshot(self, primary_interval):
        """
        Save snapshot of current primary interval solution to individual files.
        
        Creates snapshot files for each equation showing the spatial solution at
        the current primary interval. Files are saved as CSV format in a snapshots
        subdirectory for easy monitoring during long simulations.
        
        Args:
            primary_interval (int): Index of the current primary interval.
            
        Returns:
            None
            
        Side Effects:
            - Creates snapshots directory if it doesn't exist
            - Saves CSV files for each equation with spatial grid and solution data
            - Files named as 'equation_{eqn}_primary_{interval}.csv'
            
        Notes:
            Files contain two columns: spatial coordinate and solution value.
            Safe to call during simulation without interfering with main HDF5 output.
            Initial snapshot (primary_interval=0) saved for reference.
        """
        
        # Create snapshots directory if it doesn't exist
        snapshots_dir = os.path.join(self.directory_name, 'snapshots')
        if not os.path.exists(snapshots_dir):
            os.makedirs(snapshots_dir)
        
        # Save snapshot for each equation
        for eqn in range(self.num_ODEs):
            filename = f'equation_{eqn}_primary_{primary_interval:04d}.csv'
            filepath = os.path.join(snapshots_dir, filename)
            
            # Get current solution data
            spatial_coords = self.xs
            solution_values = self.y[eqn, :, primary_interval]
            
            # Write to CSV file
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(['spatial_coordinate', f'equation_{eqn}_solution'])
                # Write data rows
                for x, y_val in zip(spatial_coords, solution_values):
                    writer.writerow([x, y_val])
        
        #if self.verbose:
        #    print(f"Saved snapshots for primary interval {primary_interval} to {snapshots_dir}")

    def save_runtime_data(self, primary_interval, primary_runtime, total_runtime):
        """
        Save runtime performance data for the current primary interval.
        
        Creates a text file containing detailed timing information for each completed
        primary interval, including individual component times, cumulative runtime,
        and estimated remaining time.
        
        Args:
            primary_interval (int): Index of the completed primary interval.
            primary_runtime (dict): Dictionary containing timing information for this
                primary interval with keys:
                - 'total': Total time for primary interval
                - 'BldTPoly': Time spent building Taylor polynomials  
                - 'IntegrateGij': Time spent in integration
                - 'update_save': Time spent saving data
                - 'snapshot_save': Time spent saving snapshots
                - 'nan_check': Time spent checking for NaN values
            total_runtime (float): Total simulation runtime elapsed so far.
            
        Returns:
            None
            
        Side Effects:
            - Creates runtime_data directory if it doesn't exist
            - Saves text file with timing information
            - Updates internal runtime tracking arrays
            
        Notes:
            Files are named 'runtime_primary_{interval:04d}.txt' for easy sorting.
            Estimated remaining time uses an exponentially weighted moving average
            of recent primary interval times for better accuracy.
        """
        
        # Create runtime data directory if it doesn't exist
        runtime_dir = os.path.join(self.directory_name, 'runtime_data')
        if not os.path.exists(runtime_dir):
            os.makedirs(runtime_dir)
        
        # Store this primary's runtime for moving average calculation
        self.primary_runtimes.append(primary_runtime['total'])
        
        # Calculate estimated remaining time using moving average of last few primaries
        primaries_remaining = self.n - (primary_interval + 1)
        if len(self.primary_runtimes) >= 3:
            # Use exponentially weighted moving average of last 5 primaries
            recent_runtimes = self.primary_runtimes[-5:]
            weights = np.exp(np.linspace(-1, 0, len(recent_runtimes)))
            weights = weights / np.sum(weights)
            avg_primary_time = np.average(recent_runtimes, weights=weights)
        elif len(self.primary_runtimes) >= 2:
            # Use simple average of available data
            avg_primary_time = np.mean(self.primary_runtimes[-3:])
        else:
            # Use current primary time as estimate
            avg_primary_time = primary_runtime['total']
        
        estimated_remaining = avg_primary_time * primaries_remaining
        
        # Format the runtime data
        filename = f'runtime_primary_{primary_interval+1:04d}.txt'
        filepath = os.path.join(runtime_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(f"Runtime Data for Primary Interval {primary_interval + 1}\n")
            f.write("=" * 50 + "\n\n")
            
            # Primary interval timing breakdown
            f.write("Primary Interval Timing Breakdown:\n")
            f.write("-" * 35 + "\n")
            f.write(f"Total primary interval time:  {primary_runtime['total']:.4f} seconds\n")
            f.write(f"  BldTPoly time:              {primary_runtime['BldTPoly']:.4f} seconds\n")
            f.write(f"  IntegrateGij time:          {primary_runtime['IntegrateGij']:.4f} seconds\n")
            f.write(f"  Update/save time:           {primary_runtime['update_save']:.4f} seconds\n")
            f.write(f"  Snapshot save time:         {primary_runtime['snapshot_save']:.4f} seconds\n")
            f.write(f"  NaN check time:             {primary_runtime['nan_check']:.4f} seconds\n\n")
            
            # Overall simulation progress
            f.write("Overall Simulation Progress:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total simulation runtime:     {total_runtime:.4f} seconds\n")
            f.write(f"Completed primary intervals:  {primary_interval + 1} / {self.n}\n")
            f.write(f"Progress:                     {100 * (primary_interval + 1) / self.n:.1f}%\n")
            f.write(f"Primaries remaining:          {primaries_remaining}\n\n")
            
            # Time estimates
            f.write("Time Estimates:\n")
            f.write("-" * 16 + "\n")
            f.write(f"Average primary time:         {avg_primary_time:.4f} seconds\n")
            f.write(f"Estimated remaining time:     {estimated_remaining:.2f} seconds\n")
            
            # Convert to more readable units if needed
            if estimated_remaining > 3600:
                hours = estimated_remaining / 3600
                f.write(f"                              {hours:.2f} hours\n")
            elif estimated_remaining > 60:
                minutes = estimated_remaining / 60
                f.write(f"                              {minutes:.2f} minutes\n")
            
            # Estimated completion time
            current_time = time.time()
            estimated_completion = current_time + estimated_remaining
            completion_str = time.strftime('%Y-%m-%d %H:%M:%S', 
                                        time.localtime(estimated_completion))
            f.write(f"Estimated completion:         {completion_str}\n\n")
            
            # Performance metrics
            if len(self.primary_runtimes) > 1:
                f.write("Performance Metrics:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Fastest primary time:         {min(self.primary_runtimes):.4f} seconds\n")
                f.write(f"Slowest primary time:         {max(self.primary_runtimes):.4f} seconds\n")
                f.write(f"Average primary time:         {np.mean(self.primary_runtimes):.4f} seconds\n")
                f.write(f"Standard deviation:           {np.std(self.primary_runtimes):.4f} seconds\n")
                
                # Performance trend (comparing first half to second half)
                if len(self.primary_runtimes) >= 4:
                    mid_point = len(self.primary_runtimes) // 2
                    first_half_avg = np.mean(self.primary_runtimes[:mid_point])
                    second_half_avg = np.mean(self.primary_runtimes[mid_point:])
                    trend = ((second_half_avg - first_half_avg) / first_half_avg) * 100
                    if abs(trend) > 5:  # Only report significant trends
                        trend_word = "slower" if trend > 0 else "faster"
                        f.write(f"Performance trend:            {abs(trend):.1f}% {trend_word} recently\n")
        
        #if self.verbose:
        #    print(f"Saved runtime data for primary interval {primary_interval + 1} to {runtime_dir}")

    def set_boundary_values(self, array, index=0, time_index=None, current_time=None):
        """
        Set boundary values for a given array and equation index.
        
        Applies specified boundary conditions (fixed, open, Neumann, or time-dependent)
        to either 2D or 3D arrays. Handles both single-equation and multi-equation cases.
        
        Args:
            array (np.ndarray): The array to set boundary values for. Can be 2D (spatial)
                or 3D (spatial + time).
            index (int, optional): The equation index in multi-equation systems. 
                Defaults to 0.
            time_index (int, optional): The time index for 3D arrays. If None, treats
                as 2D array. Defaults to None.
            current_time (float, optional): Current simulation time for time-dependent
                boundaries. Defaults to None (uses 0.0).
        
        Returns:
            None
        
        Notes:
            Boundary types supported:
            - 'fixed': Use specified constant value
            - 'open': Extrapolate from interior (2nd order)
            - 'reflecting': Enforces du/dx = 0 -> u[0] = u[1]; u[-1] = u[-2]
            - 'neumann': Apply specified derivative condition
            - 'time_dependent': Evaluate user-provided function
            
            For open boundaries: u[0] = 2*u[1] - u[2] (left), u[-1] = 2*u[-2] - u[-3] (right)
        """
        if time_index is not None:
            target_array = array[index, :, time_index]
        else:
            target_array = array[index]

        ## Add debug prints 9/12
        #if time_index is not None:
        #    print(f"Before BC: left={target_array[0]:.3f}, right={target_array[-1]:.3f}")

        
        # Left boundary
        if self.boundary_type[index][0] == 'fixed':
            target_array[0] = self.boundary_vals[index][0]
        elif self.boundary_type[index][0] == 'open':
            target_array[0] = 2 * target_array[1] - target_array[2]
        elif self.boundary_type[index][0] == 'reflecting':
            target_array[0] = target_array[1]
        elif self.boundary_type[index][0] == 'neumann':
            target_array[0] = (4 * target_array[1] - target_array[2] - 2 * self.boundary_vals[index][0]) / 3
        elif self.boundary_type[index][0] == 'time_dependent':
            if current_time is None:
                current_time = 0.0  # Use 0.0 as default time if not provided
            target_array[0] = self.time_dependent_funcs[index][0](current_time, self.r_values[0])  # Pass 'r' value # 10/9 addition
        
        # Right boundary
        if self.boundary_type[index][1] == 'fixed':
            target_array[-1] = self.boundary_vals[index][1]
        elif self.boundary_type[index][1] == 'open':
            target_array[-1] = 2 * target_array[-2] - target_array[-3]
        elif self.boundary_type[index][1] == 'reflecting':
            #print(f"Applying reflecting BC: right boundary")
            target_array[-1] = target_array[-2]
        elif self.boundary_type[index][1] == 'neumann':
            target_array[-1] = (4 * target_array[-2] - target_array[-3] + 2 * self.boundary_vals[index][1]) / 3
        elif self.boundary_type[index][1] == 'time_dependent':
            if current_time is None:
                current_time = 0.0  # Use 0.0 as default time if not provided
            target_array[-1] = self.time_dependent_funcs[index][1](current_time, self.r_values[-1])  # Pass 'r' value # 10/9 addition

        ## Add debug prints 9/12
        #if time_index is not None:
        #    print(f"After BC: left={target_array[0]:.3f}, right={target_array[-1]:.3f}")
        
        if time_index is not None:
            array[index, :, time_index] = target_array
        else:
            array[index] = target_array

    def check_for_nan_results(self, primary_interval):
        """
        Check for NaN values in simulation results and terminate if found.
        
        Scans the current primary interval solution for any NaN or infinite values
        that indicate numerical instability. Provides early termination to save
        computational resources when simulations become unstable.
        
        Args:
            primary_interval (int): Index of the current primary interval.
            
        Returns:
            bool: True if simulation should continue, False if NaN detected.
            
        Side Effects:
            - Prints error message if NaN detected
            - Prints location information (equation index) for debugging
            
        Notes:
            Checks both NaN and infinite values using numpy functions.
            Called after each primary interval completion for early detection.
            Allows graceful termination with proper cleanup and timing information.
        """
        
        # Check each equation for NaN or infinite values
        for eqn in range(self.num_ODEs):
            current_solution = self.y[eqn, :, primary_interval]
            
            # Check for NaN values
            if np.any(np.isnan(current_solution)):
                print(f"Error: NaN results detected in equation {eqn} on primary {primary_interval}")
                return False
            
            # Check for infinite values
            if np.any(np.isinf(current_solution)):
                print(f"Error: Infinite results detected in equation {eqn} on primary {primary_interval}")
                return False
            
            # Optional: Check for extremely large values that might indicate instability
            max_val = np.max(np.abs(current_solution))
            if max_val > 1e15:  # Adjustable threshold
                print(f"Warning: Very large values detected in equation {eqn} on primary {primary_interval} (max = {max_val:.2e})")
                # Don't terminate for large values, just warn
        
        return True  # Continue simulation
    
    def integratePDEs(self):
        """
        Enhanced version of integratePDEs with snapshot saving, NaN detection, and runtime tracking.
        
        Integrates the system of PDEs over all primary intervals with added
        monitoring capabilities for long-running simulations. Includes automatic
        snapshot saving, runtime tracking, and early termination on numerical instability.
        
        Returns:
            None
        
        Side Effects:
            - Updates self.y with solutions at all completed primary intervals
            - Saves snapshots after each primary interval
            - Saves runtime data after each primary interval
            - Saves progress to HDF5 file after each primary interval
            - Terminates early if NaN values detected
            - Prints timing and status information if verbose mode enabled
        
        Notes:
            Enhanced version of the original integratePDEs method with:
            1. Initial snapshot at t=0
            2. Snapshot saving after each primary interval
            3. Runtime tracking and performance analysis
            4. NaN detection and early termination
            5. Proper cleanup and final state saving on early termination
        """
        # Initialize runtime tracking
        self.simulation_start_time = time.time()
        
        self.initial_save_h5()
        
        # Save initial snapshot at t=0
        self.save_primary_snapshot(0)
        
        for i in range(self.n):
            self.current_primary = i
            t_start = i * self.n**(self.k-1) * self.hbar
            
            # Timing for this primary interval
            primary_start_time = time.time()
            time_0 = primary_start_time
            
            self.BldTPoly(t_start)
            time_1 = time.time()
            
            self.IntegrateGij(t_start, i)
            time_2 = time.time()
            
            self.update_save_h5(i)
            time_3 = time.time()
            
            # Save snapshot after completing this primary interval
            self.save_primary_snapshot(i + 1)
            time_4 = time.time()
            
            # Check for NaN results and terminate if found
            if not self.check_for_nan_results(i + 1):
                print(f"Simulation terminated due to numerical instability on primary {i + 1}")
                # Save final state before terminating
                with h5.File(self.directory_name + 'output.hdf5', 'r+') as f:
                    fields = f['fields']
                    fields[:, :, i + 1] = self.y[:, :, i + 1]
                    f.attrs['complete_flag'] = False  # Mark as incomplete
                    f.attrs['termination_reason'] = 'numerical_instability'
                    f.attrs['final_primary'] = i + 1
                break
            
            time_5 = time.time()
            
            # Prepare runtime data for this primary interval
            primary_runtime = {
                'total': time_5 - time_0,
                'BldTPoly': time_1 - time_0,
                'IntegrateGij': time_2 - time_1,
                'update_save': time_3 - time_2,
                'snapshot_save': time_4 - time_3,
                'nan_check': time_5 - time_4
            }
            
            total_runtime = time_5 - self.simulation_start_time
            
            # Save runtime data
            self.save_runtime_data(i, primary_runtime, total_runtime)
            time_6 = time.time()
            
            if self.verbose:
                print("\n")
                print(f"Completed primary interval {i+1} out of {self.n}.")
                print("Total primary interval time: " + str(time_6-time_0))
                print("BldTPoly time: " + str(time_1-time_0))
                print("IntegrateGij time: " + str(time_2-time_1))
                print("Other processes time: " + str(time_6-time_2))
                print("\n")
        else:
            # This executes only if the loop completed normally (no break)
            # Explicitly save the final state to ensure it's captured
            with h5.File(self.directory_name + 'output.hdf5', 'r+') as f:
                fields = f['fields']
                fields[:, :, self.n] = self.y[:, :, self.n]
                f.attrs['complete_flag'] = True
            
            # Print final summary
            final_runtime = time.time() - self.simulation_start_time
            if self.verbose:
                print("\nSimulation completed successfully")
                print(f"Total simulation runtime: {final_runtime:.2f} seconds")
                if final_runtime > 60:
                    print(f"                          {final_runtime/60:.2f} minutes")
                if final_runtime > 3600:
                    print(f"                          {final_runtime/3600:.2f} hours")
        
    def BldTPoly(self, t_start):
        """
        Construct Taylor polynomial coefficient arrays for all secondary intervals.
        
        Loops over all secondary subintervals within the current primary interval
        to build the coefficient matrix used in time integration. Each secondary
        interval contributes one set of coefficients to the self.ll array.
        
        Args:
            t_start (float): Starting time of the current primary interval.
        
        Returns:
            None
        
        Side Effects:
            - Updates self.ll array with Taylor polynomial coefficients
            - Updates self.U through NextInCond() calls
            - Optionally saves secondary interval data
        
        Notes:
            For each secondary interval:
            1. Derivs(): Calculate function and derivative values
            2. BldTMat(): Build coefficient matrix
            3. Store coefficients in self.ll[:, :, :, i]
            4. NextInCond(): Advance to next secondary interval
        """
        for i in range(self.n**(self.k-1)):
            self.current_secondary = i
            self.Derivs()
            self.BldTMat()
            self.ll[:, :, :, i] = self.mat
            self.update_save_secondary_h5(self.current_primary, i)  # Save at current secondary if needed
            self.NextInCond()
    
    def Derivs(self):
        """
        Calculate auxiliary functions, driver functions, and their derivatives.
        
        Evaluates all auxiliary functions and driver function derivatives up to order r
        at the current secondary subinterval using lambda functions from the symbolic
        equation processing system.
        
        Returns:
            None
        
        Side Effects:
            - Updates self.aux with auxiliary function values
            - Updates self.f_array with driver functions and derivatives
            - Applies boundary conditions to self.U
        
        Notes:
            Evaluation order:
            1. Apply current boundary conditions
            2. Evaluate auxiliary functions at all grid points
            3. Evaluate main functions and derivatives at interior points
            4. Apply open boundary conditions to derivatives if needed
            
            Results are rounded to 12 decimal places for numerical stability.
        """
        gridpoint_indices = np.arange(1, self.N-1)
        gridpoint_indices_aux = np.arange(self.N)
        
        # Create base argument list for lambda functions
        base_arg_list = [self.r_values]  # Include 'r' values for all points
        
        # Auxiliary functions:
        for aux2 in range(self.num_aux):
            base_arg_list.append(self.aux[aux2])
        
        # Functions and function derivatives:
        for eqn in range(self.num_ODEs):
            base_arg_list.append(self.U[eqn,:])
            for tylr in range(self.r):
                base_arg_list.append(self.f_array[eqn,tylr,:])

        # Create aux_arg_list and eqn_arg_list
        aux_arg_list = [gridpoint_indices_aux] + base_arg_list
        eqn_arg_list = [gridpoint_indices] + base_arg_list

        # Boundary step
        current_time = self.hbar * (self.n**(self.k-1) * self.current_primary + self.current_secondary)
        if self.num_ODEs == 1:
            self.set_boundary_values(self.U, current_time=current_time)
        else:
            for i in range(self.num_ODEs):
                self.set_boundary_values(self.U, index=i, current_time=current_time)

        # Tolerance for rounding computations
        tolerance = 12  # using np.round, tolerance is an integer for the number of decimals to round to past the decimal

        # Auxiliary function evaluation
        for auxiliary in range(self.num_aux):
            self.aux[auxiliary] = self.evals.d_dict['d_'+str(auxiliary)+'_lambdified'](*aux_arg_list)
            self.aux[auxiliary] = np.round(self.aux[auxiliary], tolerance)

        # Evaluate main functions and derivatives
        for eqn in range(self.num_ODEs):
            for tylr in range(self.r+1):
                self.f_array[eqn,tylr,gridpoint_indices] = self.evals.f_dict['f'+str(eqn)+'_'+str(tylr)+'_lambdified'](*eqn_arg_list)
                self.f_array[eqn,tylr,gridpoint_indices] = np.round(self.f_array[eqn,tylr,gridpoint_indices], tolerance)
                
                # Apply open boundary conditions if necessary
                if self.boundary_type[eqn][0] == 'open':
                    self.f_array[eqn,tylr,0] = 2*self.f_array[eqn,tylr,1] - self.f_array[eqn,tylr,2]
                elif self.boundary_type[eqn][0] == 'reflecting':
                    self.f_array[eqn,tylr,0] = self.f_array[eqn,tylr,1]
                if self.boundary_type[eqn][1] == 'open':
                    self.f_array[eqn,tylr,-1] = 2*self.f_array[eqn,tylr,-2] - self.f_array[eqn,tylr,-3]
                elif self.boundary_type[eqn][1] == 'reflecting':
                    self.f_array[eqn,tylr,-1] = self.f_array[eqn,tylr,-2]

    def BldTMat(self):
        """
        Build the Taylor polynomial coefficient matrix.
        
        Constructs the coefficient matrix for Taylor polynomial representation
        by combining current solution values with computed driver function derivatives.
        
        Returns:
            None
        
        Side Effects:
            Updates self.mat with structure:
            - mat[:, 0, :]: Current solution values U (interior points only)
            - mat[:, 1:, :]: Driver function derivatives f_array (orders 0 to r)
        
        Notes:
            The coefficient matrix enables Taylor series expansion:
            u(t+Δt) ≈ Σ(i=0 to r) mat[:, i, :] * (Δt)^i / i!
            
            Only interior points (indices 1 to N-2) are included since boundary
            conditions are handled separately.
        """
        self.mat[:, 0, :] = self.U[:, 1:-1]
        self.mat[:, 1:, :] = self.f_array[:, :, 1:-1]

    def NextInCond(self):
        """
        Calculate initial conditions for the next secondary subinterval.
        
        Uses Taylor polynomial coefficients to approximate solution values at the
        start of the next secondary interval. This provides the initial conditions
        for continuing the integration process.
        
        Returns:
            None
        
        Side Effects:
            Updates self.U[:, 1:-1] with predicted values for the next time step.
        
        Notes:
            Implements Taylor series advancement:
            u(t+h) = Σ(i=0 to r+1) mat[:, i, :] * h^i / i!
            
            where h = self.hbar is the secondary time step.
            
            Only updates interior points; boundary conditions are applied
            separately in the Derivs() method.
        """
        powers = np.arange(self.r + 2)
        coeffs = (self.hbar ** powers) / self.factorials
        self.U[:, 1:-1] = np.sum(self.mat * coeffs[:, np.newaxis], axis=1)

    @abstractmethod
    def IntegrateGij(self, t_start, primary_interval):
        """
        Integrate PDEs over a primary subinterval.
        
        Abstract method that must be implemented by concrete simulation classes
        to perform the actual time integration using method-specific algorithms
        (explicit or implicit finite difference schemes).
        
        Args:
            t_start (float): Starting time of the current primary subinterval.
            primary_interval (int): Index of the current primary subinterval.
        
        Returns:
            None
        
        Notes:
            Concrete implementations typically:
            1. Loop over secondary intervals within the primary
            2. Perform numerical integration using stored Taylor coefficients
            3. Handle custom output time interpolation
            4. Update solution arrays for the next primary interval
            
            The integration method (explicit vs implicit) determines the specific
            algorithm used for advancing the solution in time.
        """
        pass
