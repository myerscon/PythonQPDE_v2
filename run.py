"""
Main execution script for the quantum PDE solver.

This module provides the command-line interface for running PDE simulations with
specified input files. Handles input script loading, parameter validation, solver
selection, and execution timing.
"""

import argparse
import importlib.util
import os
import sys
import time
from solvers.explicit_fdm_solvers import Explicit_fdm_simulation, Explicit_vectorized_fdm_simulation
from solvers.implicit_fdm_solvers import Implicit_fdm_simulation, Implicit_vectorized_fdm_simulation

def load_input_script(script_path):
    """ Load input script from command line argument.
    
    Dynamically imports the specified input script file to access simulation
    parameters and equation definitions.
    
    Args:
        script_path (str): Path to the input script file.
        
    Returns:
        module: Loaded input script module containing general_param_dict and
            sympy_param_dict.
            
    Notes:
        Uses importlib to dynamically load the script and add it to sys.modules
        for proper module handling during execution.
    """
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    input_script = importlib.util.module_from_spec(spec)
    sys.modules[script_name] = input_script
    spec.loader.exec_module(input_script)
    return input_script

def main():
    """ Main execution function for PDE solver.
    
    Parses command line arguments, loads the specified input script, selects
    appropriate solver based on time discretization method, and executes the
    simulation with timing information.
    
    Command line usage:
        python run.py input_script.py [--output-dir OUTPUT_DIR]
        
    Notes:
        Automatically selects between explicit and implicit solvers based on
        time_discretization parameter. Chooses vectorized or non-vectorized
        implementation based on use_vectorized parameter. Prints simulation
        parameters and runtime statistics.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run PDE solver with specified input script.")
    parser.add_argument('input_script', type=str, help="Path to the input script")
    parser.add_argument('--output-dir', '-o', type=str, help="Override output directory")
    args = parser.parse_args()

    # Load the input script
    input_script = load_input_script(args.input_script)

    # Override output directory if specified
    if args.output_dir:
        input_script.general_param_dict['directory_name'] = args.output_dir
        print(f"Output directory overridden to: {args.output_dir}")

    # Run simulation using specified input script
    if input_script.general_param_dict['time_discretization'] == 'explicit':
        if input_script.general_param_dict.get('use_vectorized', True):
            my_sim = Explicit_vectorized_fdm_simulation(input_script.general_param_dict, input_script.sympy_param_dict)
        else:
            my_sim = Explicit_fdm_simulation(input_script.general_param_dict, input_script.sympy_param_dict)
    elif input_script.general_param_dict['time_discretization'] in ['implicit', 'implicit_pick_nk']:
        if input_script.general_param_dict.get('use_vectorized', True):
            my_sim = Implicit_vectorized_fdm_simulation(input_script.general_param_dict, input_script.sympy_param_dict)
        else:
            my_sim = Implicit_fdm_simulation(input_script.general_param_dict, input_script.sympy_param_dict)

    # Print simulation parameters
    print("\nSIMULATION PARAMETERS")
    print("Total simulation time T:", str(my_sim.hbar * my_sim.n**my_sim.k))
    print("dt calculated:", str(my_sim.hbar))
    print("number of primary intervals (n):", str(my_sim.n))
    print("k:", str(my_sim.k))
    print("Number of secondaries per primary (n**(k-1)):", str(my_sim.n**(my_sim.k-1)))
    print("Output directory:", input_script.general_param_dict['directory_name'])
    print()

    # Run simulation and time it
    start = time.time()
    my_sim.integratePDEs()
    end = time.time()
    print("Total runtime:", str(end-start), "seconds")
    

if __name__ == "__main__":
    main()