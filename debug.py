"""
Debugging utilities for the quantum PDE solver.

This module provides specialized debugging tools for analyzing simulation initialization,
secondary time stepping, and primary interval integration. Includes plotting capabilities
for visualizing solution evolution and equation state at different stages.
"""

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from solvers.explicit_fdm_solvers import Explicit_fdm_simulation, Explicit_vectorized_fdm_simulation
from solvers.implicit_fdm_solvers import Implicit_fdm_simulation, Implicit_vectorized_fdm_simulation
from pprint import pformat

def load_input_script(script_path):
    """ Load input script from command line argument.
    
    Dynamically imports the specified input script file for debugging analysis.
    
    Args:
        script_path (str): Path to the input script file.
        
    Returns:
        module: Loaded input script module containing simulation parameters.
    """
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    input_script = importlib.util.module_from_spec(spec)
    sys.modules[script_name] = input_script
    spec.loader.exec_module(input_script)
    return input_script

def save_object_data(obj, filename):
    """ Save object attributes to a text file.
    
    Extracts non-callable, non-private attributes from an object and writes
    them to a formatted text file for debugging inspection.
    
    Args:
        obj: Object to inspect and save.
        filename (str): Output file path for saved data.
    """
    with open(filename, 'w') as f:
        attrs = {k: v for k, v in vars(obj).items() 
                if not callable(v) and not k.startswith('_')}
        f.write(pformat(attrs, indent=4))

def save_equation_writer_data(eqn_writer, filename):
    """ Save equation writer object data including function and coefficient dictionaries.
    
    Exports the equation writer's d_dict, f_dict, and c_dict (if available) to
    a formatted text file for debugging equation processing.
    
    Args:
        eqn_writer: Equation writer object to inspect.
        filename (str): Output file path for saved data.
        
    Notes:
        Includes c_dict for implicit solvers when available. Provides detailed
        view of lambda functions and symbolic expressions.
    """
    with open(filename, 'w') as f:
        f.write("Equation Writer Object Data\n")
        f.write("=========================\n\n")
        f.write("Function Dictionary (d_dict):\n")
        f.write("--------------------------\n")
        f.write(pformat(eqn_writer.d_dict, indent=4))
        f.write("\n\n")
        
        # Check if c_dict exists (for implicit solvers)
        if hasattr(eqn_writer, 'c_dict'):
            f.write("Coefficient Dictionary (c_dict):\n")
            f.write("---------------------------\n")
            f.write(pformat(eqn_writer.c_dict, indent=4))
            f.write("\n\n")
            
        f.write("Function Dictionary (f_dict):\n")
        f.write("--------------------------\n")
        f.write(pformat(eqn_writer.f_dict, indent=4))

def create_debug_directory(input_file_path, mode):
    """ Create a debug output directory for the specified mode.
    
    Creates timestamped directory for organizing debug outputs based on
    input file location and debug mode.
    
    Args:
        input_file_path (str): Path to input file being debugged.
        mode (str): Debug mode ('init', 'secondary', 'primary').
        
    Returns:
        Path: Created debug directory path.
    """
    base_dir = Path(os.path.dirname(input_file_path))
    debug_dir = base_dir / f"debug_{mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir

def plot_variables_initial(sim, filename=None, plot_alpha=False, plot_alpha_dot=False):
    """ Create initial condition plots for all variables and auxiliaries.
    
    Generates subplot visualization of initial conditions for main variables,
    auxiliary functions, and optionally alpha/alpha_dot expressions.
    
    Args:
        sim: Simulation object with initialized conditions.
        filename (str, optional): Output file path. Defaults to None.
        plot_alpha (bool, optional): Include alpha plot. Defaults to False.
        plot_alpha_dot (bool, optional): Include alpha_dot plot. Defaults to False.
        
    Notes:
        Creates one subplot per variable type. Alpha plots assume alpha is
        derived from the first variable for demonstration purposes.
    """
    num_vars = sim.num_ODEs
    num_aux = sim.num_aux
    total_plots = num_vars + num_aux
    
    # Add extra plots for alpha and alpha_dot if requested
    extra_plots = sum([plot_alpha, plot_alpha_dot])
    if total_plots + extra_plots > 0:
        fig, axes = plt.subplots(1, total_plots + extra_plots, figsize=(8*(total_plots + extra_plots), 6))
        if total_plots + extra_plots == 1:
            axes = [axes]
        
        # Plot main variables
        for i in range(num_vars):
            axes[i].plot(sim.xs, sim.U[i], label=f'Variable {i}')
            axes[i].set_xlabel('x')
            axes[i].set_ylabel(f'Variable {i}')
            axes[i].grid(True)
            axes[i].legend()
        
        # Plot auxiliary variables
        for i in range(num_aux):
            idx = i + num_vars
            axes[idx].plot(sim.xs, sim.aux[i], label=f'Auxiliary {i}')
            axes[idx].set_xlabel('x')
            axes[idx].set_ylabel(f'Auxiliary {i}')
            axes[idx].grid(True)
            axes[idx].legend()
        
        # Add alpha plot if requested
        if plot_alpha:
            idx = total_plots
            alpha_initial = sim.U[0].copy()  # Assuming alpha is derived from first variable
            axes[idx].plot(sim.xs, alpha_initial, label='Alpha')
            axes[idx].set_xlabel('x')
            axes[idx].set_ylabel('Alpha')
            axes[idx].grid(True)
            axes[idx].legend()

        # Add alpha_dot plot if requested
        if plot_alpha_dot:
            idx = total_plots + plot_alpha
            alpha_dot_initial = np.zeros_like(sim.U[0])  # Initial alpha_dot is zero
            axes[idx].plot(sim.xs, alpha_dot_initial, label='Alpha Dot')
            axes[idx].set_xlabel('x')
            axes[idx].set_ylabel('Alpha Dot')
            axes[idx].grid(True)
            axes[idx].legend()
            
        plt.tight_layout()
        if filename:
            fig.savefig(filename)
            plt.close(fig)

def plot_variables_evolution(sim, states, times, filename=None, plot_alpha=False, plot_alpha_dot=False):
    """ Plot evolution of variables and auxiliaries on same subplots.
    
    Creates time series visualization showing how variables evolve over
    the specified time points during debugging analysis.
    
    Args:
        sim: Simulation object.
        states (dict): Dictionary containing variable evolution data with keys
            'vars', 'aux', and optionally 'alpha', 'alpha_dot'.
        times (list): Time points corresponding to stored states.
        filename (str, optional): Output file path. Defaults to None.
        plot_alpha (bool, optional): Include alpha evolution. Defaults to False.
        plot_alpha_dot (bool, optional): Include alpha_dot evolution. Defaults to False.
        
    Notes:
        Each subplot shows multiple time curves for one variable type.
        Useful for tracking solution behavior during time stepping.
    """
    num_vars = sim.num_ODEs
    num_aux = sim.num_aux
    total_plots = num_vars + num_aux
    
    # Add extra plots for alpha and alpha_dot if requested
    extra_plots = sum([plot_alpha, plot_alpha_dot])
    if total_plots + extra_plots > 0:
        fig, axes = plt.subplots(1, total_plots + extra_plots, figsize=(8*(total_plots + extra_plots), 6))
        if total_plots + extra_plots == 1:
            axes = [axes]
        
        # Plot main variables
        for i in range(num_vars):
            for t_idx, t in enumerate(times):
                axes[i].plot(sim.xs, states['vars'][i][t_idx], 
                           label=f't = {t:.6f}')
            axes[i].set_xlabel('x')
            axes[i].set_ylabel(f'Variable {i}')
            axes[i].grid(True)
            axes[i].legend()
        
        # Plot auxiliary variables
        for i in range(num_aux):
            idx = i + num_vars
            for t_idx, t in enumerate(times):
                axes[idx].plot(sim.xs, states['aux'][i][t_idx], 
                             label=f't = {t:.6f}')
            axes[idx].set_xlabel('x')
            axes[idx].set_ylabel(f'Auxiliary {i}')
            axes[idx].grid(True)
            axes[idx].legend()

        # Plot alpha if requested
        if plot_alpha and 'alpha' in states:
            idx = total_plots
            for t_idx, t in enumerate(times[1:], start=0):  # Skip initial time for alpha
                axes[idx].plot(sim.xs, states['alpha'][t_idx], 
                             label=f't = {t:.6f}')
            axes[idx].set_xlabel('x')
            axes[idx].set_ylabel('Alpha')
            axes[idx].grid(True)
            axes[idx].legend()

        # Plot alpha_dot if requested
        if plot_alpha_dot and 'alpha_dot' in states:
            idx = total_plots + plot_alpha
            for t_idx, t in enumerate(times[1:], start=0):  # Skip initial time for alpha_dot
                axes[idx].plot(sim.xs, states['alpha_dot'][t_idx], 
                             label=f't = {t:.6f}')
            axes[idx].set_xlabel('x')
            axes[idx].set_ylabel('Alpha Dot')
            axes[idx].grid(True)
            axes[idx].legend()
        
        plt.tight_layout()
        if filename:
            fig.savefig(filename)
            plt.close(fig)

def initialize_simulation(input_file):
    """ Initialize simulation from input file.
    
    Creates appropriate simulation object (explicit/implicit, vectorized/non-vectorized)
    based on parameters in the input file.
    
    Args:
        input_file (str): Path to input script file.
        
    Returns:
        Simulation: Initialized simulation object ready for debugging.
        
    Notes:
        Automatically selects solver type based on time_discretization and
        use_vectorized parameters from the input file.
    """
    input_script = load_input_script(input_file)
    
    if input_script.general_param_dict['time_discretization'] == 'explicit':
        if input_script.general_param_dict.get('use_vectorized', True):
            sim = Explicit_vectorized_fdm_simulation(input_script.general_param_dict, input_script.sympy_param_dict)
        else:
            sim = Explicit_fdm_simulation(input_script.general_param_dict, input_script.sympy_param_dict)
    elif input_script.general_param_dict['time_discretization'] in ['implicit', 'implicit_pick_nk']:
        if input_script.general_param_dict.get('use_vectorized', True):
            sim = Implicit_vectorized_fdm_simulation(input_script.general_param_dict, input_script.sympy_param_dict)
        else:
            sim = Implicit_fdm_simulation(input_script.general_param_dict, input_script.sympy_param_dict)
    
    return sim

def debug_initialization(args):
    """ Debug initialization phase of simulation.
    
    Analyzes the simulation setup by saving object data, equation writer
    information, and initial condition plots to a timestamped debug directory.
    
    Args:
        args: Command line arguments containing input script path and debug options.
        
    Notes:
        Provides detailed view of simulation state immediately after initialization
        but before any time stepping. Useful for verifying equation setup and
        initial conditions.
    """
    sim = initialize_simulation(args.input_script)
    output_dir = create_debug_directory(args.input_script, "init")
    
    # Save simulation object data
    save_object_data(sim, output_dir / "simulation_data.txt")
    save_equation_writer_data(sim.evals, output_dir / "equation_writer_data.txt")
    
    # Plot and save initial conditions
    plot_variables_initial(sim, 
                         output_dir / "initial_conditions.png",
                         plot_alpha=args.plot_alpha,
                         plot_alpha_dot=args.plot_alpha_dot)
    
    print(f"Debug initialization data saved to {output_dir}")

def debug_secondary_steps(args):
    """ Debug specified number of secondary time steps.
    
    Executes a limited number of secondary time steps while tracking solution
    evolution and storing state information for analysis.
    
    Args:
        args: Command line arguments containing number of steps and plotting options.
        
    Notes:
        Calls Derivs(), BldTMat(), and NextInCond() for each step while storing
        intermediate states. Useful for analyzing time stepping behavior and
        detecting issues in derivative computation.
    """
    sim = initialize_simulation(args.input_script)
    output_dir = create_debug_directory(args.input_script, "secondary")
    
    # Storage for states and times
    states = {
        'vars': [[] for _ in range(sim.num_ODEs)],
        'aux': [[] for _ in range(sim.num_aux)]
    }
    
    # Add alpha and alpha_dot storage if requested
    if args.plot_alpha:
        states['alpha'] = []
    if args.plot_alpha_dot:
        states['alpha_dot'] = []
        
    times = [0.0]  # Start with initial time
    
    # Store initial state
    for i in range(sim.num_ODEs):
        states['vars'][i].append(sim.U[i].copy())
    for i in range(sim.num_aux):
        states['aux'][i].append(sim.aux[i].copy())
    
    # Run specified number of secondary steps
    dt = sim.hbar
    for i in range(args.num_steps):
        sim.current_secondary = i
        sim.Derivs()
        sim.BldTMat()
        sim.ll[:, :, :, i] = sim.mat
        sim.NextInCond()
        
        # Store state after step
        times.append((i + 1) * dt)
        for j in range(sim.num_ODEs):
            states['vars'][j].append(sim.U[j].copy())
        for j in range(sim.num_aux):
            states['aux'][j].append(sim.aux[j].copy())
            
        # Calculate and store alpha and alpha_dot if requested
        if args.plot_alpha:
            alpha_current = sim.U[0].copy()  # Assuming alpha is derived from first variable
            states['alpha'].append(alpha_current)
            
            if args.plot_alpha_dot:
                if i > 0:
                    alpha_previous = states['alpha'][-2]
                    alpha_dot = (alpha_current - alpha_previous) / dt
                else:
                    alpha_dot = np.zeros_like(alpha_current)
                states['alpha_dot'].append(alpha_dot)
    
    # Create evolution plot
    plot_variables_evolution(sim, states, times, 
                           output_dir / "secondary_evolution.png",
                           plot_alpha=args.plot_alpha,
                           plot_alpha_dot=args.plot_alpha_dot)
    
    print(f"Completed {args.num_steps} secondary steps. Output saved to {output_dir}")

def debug_primary_interval(args):
    """ Debug one complete primary interval with evolution monitoring.
    
    Executes all secondary intervals within one primary interval while
    optionally monitoring solution evolution at specified intervals.
    
    Args:
        args: Command line arguments containing monitoring and plotting options.
        
    Notes:
        Completes a full primary interval including IntegrateGij() call.
        Can monitor evolution at regular intervals or only store final state.
        Useful for understanding primary interval integration behavior.
    """
    sim = initialize_simulation(args.input_script)
    output_dir = create_debug_directory(args.input_script, "primary")
    
    # Storage for states and times
    states = {
        'vars': [[] for _ in range(sim.num_ODEs)],
        'aux': [[] for _ in range(sim.num_aux)]
    }
    
    # Add alpha and alpha_dot storage if requested
    if args.plot_alpha:
        states['alpha'] = []
    if args.plot_alpha_dot:
        states['alpha_dot'] = []
    
    times = [0.0]  # Start with initial time
    
    # Store initial state
    for i in range(sim.num_ODEs):
        states['vars'][i].append(sim.U[i].copy())
    for i in range(sim.num_aux):
        states['aux'][i].append(sim.aux[i].copy())
    
    # Run one primary interval
    t_start = 0
    sim.current_primary = 0
    
    # If monitoring evolution requested, determine step size
    num_secondaries = sim.n**(sim.k-1)
    if args.monitor_evolution:
        step_size = max(1, num_secondaries // args.monitor_points)
    else:
        step_size = num_secondaries  # Only store final state
    
    # Loop through secondary intervals
    for i in range(num_secondaries):
        
        sim.current_secondary = i
        sim.Derivs()
        sim.BldTMat()
        
        try:
            sim.ll[:, :, :, i] = sim.mat
        except Exception as e:
            print(f"Error during assignment: {e}")
            print("Current array states:")
            print(f"sim.ll shape: {sim.ll.shape}")
            print(f"sim.mat shape: {sim.mat.shape}")
            raise
        
        sim.NextInCond()
        
        # Store states at monitoring points
        if i % step_size == 0 or i == num_secondaries - 1:
            current_time = t_start + i * sim.hbar
            times.append(current_time)
            
            for j in range(sim.num_ODEs):
                states['vars'][j].append(sim.U[j].copy())
            for j in range(sim.num_aux):
                states['aux'][j].append(sim.aux[j].copy())
                
            # Calculate and store alpha and alpha_dot if requested
            if args.plot_alpha:
                alpha_current = sim.U[0].copy()  # Assuming alpha is derived from first variable
                states['alpha'].append(alpha_current)
                
                if args.plot_alpha_dot:
                    if len(states['alpha']) > 1:
                        alpha_previous = states['alpha'][-2]
                        alpha_dot = (alpha_current - alpha_previous) / sim.hbar
                    else:
                        alpha_dot = np.zeros_like(alpha_current)
                    states['alpha_dot'].append(alpha_dot)
    
    # Integrate over primary interval
    sim.IntegrateGij(t_start, 0)
    
    # Store final state after integration
    final_time = t_start + num_secondaries * sim.hbar
    times.append(final_time)
    
    for i in range(sim.num_ODEs):
        states['vars'][i].append(sim.y[i, :, 1].copy())
    
    # Compute final auxiliary values
    sim.Derivs()  # This updates auxiliary values for the final state
    for i in range(sim.num_aux):
        states['aux'][i].append(sim.aux[i].copy())
    
    # Calculate final alpha and alpha_dot if requested
    if args.plot_alpha:
        alpha_final = sim.y[0, :, 1].copy()
        states['alpha'].append(alpha_final)
        
        if args.plot_alpha_dot:
            if len(states['alpha']) > 1:
                alpha_previous = states['alpha'][-2]
                alpha_dot = (alpha_final - alpha_previous) / sim.hbar
            else:
                alpha_dot = np.zeros_like(alpha_final)
            states['alpha_dot'].append(alpha_dot)
    
    # Create evolution plot
    plot_variables_evolution(sim, states, times, 
                           output_dir / "primary_evolution.png",
                           plot_alpha=args.plot_alpha,
                           plot_alpha_dot=args.plot_alpha_dot)
    
    print(f"Completed primary interval. Output saved to {output_dir}")

def main():
    """ Main debugging function with command line interface.
    
    Provides command line interface for different debugging modes including
    initialization analysis, secondary step debugging, and primary interval
    monitoring.
    
    Command line usage:
        python debug.py input_script.py --mode {init,secondary,primary} [options]
        
    Available modes:
        - init: Analyze initialization phase
        - secondary: Debug specified number of secondary steps  
        - primary: Debug complete primary interval with optional monitoring
        
    Notes:
        Each mode creates timestamped output directory with relevant plots
        and data files for detailed analysis of simulation behavior.
    """
    parser = argparse.ArgumentParser(description="Debug QPDE solver")
    parser.add_argument('input_script', type=str, help="Path to the input script")
    parser.add_argument('--mode', choices=['init', 'secondary', 'primary'],
                      required=True, help="Debug mode")
    parser.add_argument('--num-steps', type=int, default=1,
                      help="Number of secondary steps to run (for secondary mode)")
    parser.add_argument('--monitor-evolution', action='store_true',
                      help="Monitor evolution during primary interval")
    parser.add_argument('--monitor-points', type=int, default=10,
                      help="Number of points to monitor during primary evolution")
    parser.add_argument('--plot-alpha', action='store_true',
                      help="Include alpha plot")
    parser.add_argument('--plot-alpha-dot', action='store_true',
                      help="Include alpha dot plot")
    
    args = parser.parse_args()
    
    if args.mode == 'init':
        debug_initialization(args)
    elif args.mode == 'secondary':
        debug_secondary_steps(args)
    elif args.mode == 'primary':
        debug_primary_interval(args)

if __name__ == "__main__":
    main()