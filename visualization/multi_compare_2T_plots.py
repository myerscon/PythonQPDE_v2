"""
Multi-method plotting script for comparing 2T simulation results.

This module provides comparison plotting functionality for multiple two-temperature (2T)
simulation methods simultaneously. Supports both command-line specification and JSON
configuration files for flexible multi-method analysis with consistent styling.

Usage examples:
# Basic usage with config file
python visualization/multi_compare_2T_plots.py --config comparison_config.json

# Command line specification
python visualization/multi_compare_2T_plots.py \
    --simulation path/to/sim1.hdf5 "Method 1" \
    --simulation path/to/sim2.hdf5 "Method 2" \
    --simulation path/to/sim3.hdf5 "Method 3" \
    --output-dir comparison_plots \
    --title "Method Comparison Study"

# Create animation frames
python visualization/multi_compare_2T_plots.py --config config.json --animation
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
import json
import itertools

# Set more reliable plot styling that won't throw font warnings
plt.style.use('ggplot')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
mpl.rcParams['text.usetex'] = False  # Set to True if LaTeX is installed
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 11

# Default configuration
DEFAULT_OUTPUT_DIR = "multi_comparison_plots"
DEFAULT_TITLE = "Multi-Method 2T Simulation Comparison"
X_LABEL = "X Value"
Y_LABEL = "Unitless Temperature"
RADIATION_LABEL = "Radiation (E)"
MATTER_LABEL = "Matter (T)"

# Plot styles
RADIATION_STYLE = '-'  # Solid
MATTER_STYLE = '--'    # Dashed
LINEWIDTH = 2

# Color palette for different methods (will cycle through these)
COLOR_PALETTE = [
    'blue', 'red', 'green', 'orange', 'purple', 'brown', 
    'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow',
    'darkblue', 'darkred', 'darkgreen', 'darkorange', 'darkviolet', 'saddlebrown'
]

class SimulationData:
    """ Container for simulation data and metadata.
    
    Encapsulates HDF5 file handling, data loading, and metadata management
    for individual simulation results in multi-method comparisons.
    
    Attributes:
        filepath (str): Path to HDF5 simulation file.
        label (str): Display label for this simulation method.
        color (str): Plot color for this simulation.
        file_handle (h5py.File): Opened HDF5 file object.
        spatial_grid (np.ndarray): Spatial coordinate array.
        fields (h5py.Dataset): Primary field data.
        times (np.ndarray): Primary interval times.
        custom_fields (h5py.Dataset): Custom output field data.
        custom_times (np.ndarray): Custom output times.
    """
    
    def __init__(self, filepath, label, color):
        """ Initialize simulation data container.
        
        Args:
            filepath (str): Path to HDF5 simulation file.
            label (str): Display label for plotting.
            color (str): Color for plot lines.
        """
        self.filepath = filepath
        self.label = label
        self.color = color
        self.file_handle = None
        self.spatial_grid = None
        self.fields = None
        self.times = None
        self.custom_fields = None
        self.custom_times = None
        
    def load(self):
        """ Load the HDF5 file and extract data.
        
        Opens HDF5 file and loads spatial grid, primary fields, and custom
        output data if available.
        
        Returns:
            bool: True if successful, False if loading failed.
        """
        try:
            self.file_handle = h5py.File(self.filepath, 'r')
            self.spatial_grid = self.file_handle['spatial_grid'][:]
            
            # Load primary field data
            if 'fields' in self.file_handle:
                self.fields = self.file_handle['fields']
                self.times = self.file_handle['primary_interval_times'][:]
            
            # Load custom field data if available
            if 'custom_fields' in self.file_handle:
                self.custom_fields = self.file_handle['custom_fields']
                self.custom_times = self.file_handle['custom_output_times'][:]
                
            print(f"Loaded {self.label} from {self.filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading {self.filepath}: {e}")
            return False
    
    def close(self):
        """ Close the HDF5 file.
        
        Safely closes the HDF5 file handle to free resources.
        """
        if self.file_handle:
            self.file_handle.close()
    
    def has_custom_data(self):
        """ Check if custom output data is available.
        
        Returns:
            bool: True if both custom_fields and custom_times are available.
        """
        return self.custom_fields is not None and self.custom_times is not None

def load_config(config_file):
    """ Load configuration from JSON file.
    
    Parses JSON configuration file containing simulation definitions and
    plotting parameters for multi-method comparison.
    
    Args:
        config_file (str): Path to JSON configuration file.
        
    Returns:
        dict: Configuration dictionary with simulation list and options,
            or None if loading failed.
            
    Notes:
        Expected JSON format includes 'simulations' list with 'file' and 'label'
        entries, plus optional 'output_dir', 'title', and 'use_custom' settings.
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading config file {config_file}: {e}")
        return None

def create_example_config(filename="comparison_config.json"):
    """ Create an example configuration file.
    
    Generates a template JSON configuration file showing the expected format
    for multi-method comparison setup.
    
    Args:
        filename (str, optional): Output filename. Defaults to "comparison_config.json".
        
    Notes:
        Creates example with multiple simulation methods including implicit
        and explicit schemes with different CFL numbers for demonstration.
    """
    config = {
        "simulations": [
            {
                "file": "problems/test_2T/implicit_output.hdf5",
                "label": "Implicit Method"
            },
            {
                "file": "problems/test_2T/explicit_cfl01_output.hdf5", 
                "label": "Explicit CFL=0.1"
            },
            {
                "file": "problems/test_2T/explicit_cfl05_output.hdf5",
                "label": "Explicit CFL=0.5"
            },
            {
                "file": "problems/test_2T/explicit_cfl08_output.hdf5",
                "label": "Explicit CFL=0.8"
            }
        ],
        "output_dir": "multi_comparison_plots",
        "title": "Multi-Method 2T Simulation Comparison",
        "use_custom": True
    }
    
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Created example configuration file: {filename}")

def setup_simulation_data(simulation_list):
    """ Set up SimulationData objects for all simulations.
    
    Creates SimulationData containers for each simulation with cycling
    colors from the predefined color palette.
    
    Args:
        simulation_list (list): List of (filepath, label) tuples.
        
    Returns:
        list: List of SimulationData objects with assigned colors.
    """
    simulations = []
    color_cycle = itertools.cycle(COLOR_PALETTE)
    
    # Add all simulations with cycling colors
    for filepath, label in simulation_list:
        color = next(color_cycle)
        simulations.append(SimulationData(filepath, label, color))
    
    return simulations

def plot_comparison(simulations, output_dir, title, use_custom=True, time_indices=None):
    """ Create comparison plots for multiple simulations.
    
    Generates comparison plots showing radiation and matter fields from all
    specified simulation methods on the same plots for direct comparison.
    
    Args:
        simulations (list): List of SimulationData objects.
        output_dir (Path): Directory for saving plots.
        title (str): Title for the plots.
        use_custom (bool, optional): Use custom output times if available. Defaults to True.
        time_indices (list, optional): Time indices to plot. Defaults to all.
        
    Notes:
        Uses solid lines for radiation field and dashed lines for matter field.
        Each simulation method gets a unique color from the color palette.
        Legend placement accounts for potentially many simulation methods.
    """
    # Determine which dataset to use and find common time range
    use_custom_data = use_custom and all(sim.has_custom_data() for sim in simulations)
    
    if use_custom_data:
        print("Using custom output data for comparison")
        # Find minimum number of custom times across all simulations
        max_times = min(sim.custom_times.shape[0] for sim in simulations)
        dataset_name = "custom"
    else:
        print("Using primary field data for comparison")
        # Find minimum number of primary times across all simulations
        max_times = min(sim.times.shape[0] for sim in simulations)
        dataset_name = "primary"
    
    # Determine which time indices to plot
    if time_indices is None:
        time_indices = range(max_times)
    else:
        # Filter out indices that are out of range
        time_indices = [i for i in time_indices if i < max_times]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots for each time index
    for time_idx in time_indices:
        plt.figure(figsize=(12, 8))
        
        # Plot each simulation
        for sim in simulations:
            if use_custom_data:
                fields = sim.custom_fields
            else:
                fields = sim.fields
            
            # Plot radiation field (field 0)
            plt.plot(sim.spatial_grid, fields[0, :, time_idx], 
                    color=sim.color, linestyle=RADIATION_STYLE, 
                    linewidth=LINEWIDTH, 
                    label=f"{sim.label} {RADIATION_LABEL}")
            
            # Plot matter field (field 1)
            plt.plot(sim.spatial_grid, fields[1, :, time_idx], 
                    color=sim.color, linestyle=MATTER_STYLE, 
                    linewidth=LINEWIDTH, 
                    label=f"{sim.label} {MATTER_LABEL}")
        
        # Use the first simulation's time for the title
        if use_custom_data:
            current_time = simulations[0].custom_times[time_idx]
        else:
            current_time = simulations[0].times[time_idx]
        
        # Add labels and title
        plt.xlabel(X_LABEL)
        plt.ylabel(Y_LABEL)
        plt.title(f"{title} (t = {current_time:.6f})")
        
        # Adjust legend to handle potentially many entries
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        # Save plot
        plt.savefig(output_dir / f'{dataset_name}_comparison_t{time_idx:04d}.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
    
    print(f"Created {len(time_indices)} comparison plots in {output_dir}")

def create_animation_frames(simulations, output_dir, title, skip=1, use_custom=True):
    """ Create frames for animation with consistent axis scaling.
    
    Generates animation frames showing evolution of all simulation methods
    with consistent y-axis scaling across all frames.
    
    Args:
        simulations (list): List of SimulationData objects.
        output_dir (Path): Directory for saving animation frames.
        title (str): Title for animation frames.
        skip (int, optional): Frame skip interval. Defaults to 1.
        use_custom (bool, optional): Use custom output times. Defaults to True.
        
    Notes:
        Pre-computes global min/max across all simulations for consistent scaling.
        Saves numbered frames suitable for ffmpeg animation creation.
    """
    # Determine which dataset to use
    use_custom_data = use_custom and all(sim.has_custom_data() for sim in simulations)
    
    if use_custom_data:
        max_times = min(sim.custom_times.shape[0] for sim in simulations)
        print("Using custom output data for animation")
    else:
        max_times = min(sim.times.shape[0] for sim in simulations)
        print("Using primary field data for animation")
    
    time_indices = range(0, max_times, skip)
    
    # Create animation frames directory
    frames_dir = output_dir / 'animation_frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Find global min and max values for consistent scaling
    global_min = float('inf')
    global_max = float('-inf')
    
    for sim in simulations:
        if use_custom_data:
            fields = sim.custom_fields
        else:
            fields = sim.fields
        
        sim_min = min(np.min(fields[0, :, :max_times]), np.min(fields[1, :, :max_times]))
        sim_max = max(np.max(fields[0, :, :max_times]), np.max(fields[1, :, :max_times]))
        
        global_min = min(global_min, sim_min)
        global_max = max(global_max, sim_max)
    
    # Add 10% padding to y-axis range
    y_range = global_max - global_min
    y_min = global_min - 0.1 * y_range
    y_max = global_max + 0.1 * y_range
    
    # Create frames
    for i, time_idx in enumerate(time_indices):
        plt.figure(figsize=(12, 8))
        
        # Plot each simulation
        for sim in simulations:
            if use_custom_data:
                fields = sim.custom_fields
            else:
                fields = sim.fields
            
            # Plot radiation field (field 0)
            plt.plot(sim.spatial_grid, fields[0, :, time_idx], 
                    color=sim.color, linestyle=RADIATION_STYLE, 
                    linewidth=LINEWIDTH, 
                    label=f"{sim.label} {RADIATION_LABEL}")
            
            # Plot matter field (field 1)
            plt.plot(sim.spatial_grid, fields[1, :, time_idx], 
                    color=sim.color, linestyle=MATTER_STYLE, 
                    linewidth=LINEWIDTH, 
                    label=f"{sim.label} {MATTER_LABEL}")
        
        # Use the first simulation's time for the title
        if use_custom_data:
            current_time = simulations[0].custom_times[time_idx]
        else:
            current_time = simulations[0].times[time_idx]
        
        # Add labels and title
        plt.xlabel(X_LABEL)
        plt.ylabel(Y_LABEL)
        plt.title(f"{title} (t = {current_time:.6f})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        # Set consistent y-axis limits
        plt.ylim(y_min, y_max)
        
        # Save frame with zero-padded frame number
        plt.savefig(frames_dir / f'frame_{i:04d}.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    print(f"Created {len(time_indices)} animation frames in {frames_dir}")
    print("You can create an animation using ffmpeg with a command like:")
    print(f"ffmpeg -framerate 10 -i {frames_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 23 {output_dir}/animation.mp4")

def main():
    """ Main function providing command line interface for multi-method comparison.
    
    Supports both JSON configuration files and direct command line specification
    of simulation files for flexible multi-method comparison plotting.
    
    Command line usage:
        python multi_compare_2T_plots.py --config config.json
        
        OR
        
        python multi_compare_2T_plots.py \\
            --simulation file1.hdf5 "Method 1" \\
            --simulation file2.hdf5 "Method 2" \\
            --simulation file3.hdf5 "Method 3"
    
    Options:
        --config: JSON configuration file
        --simulation: Simulation file and label pairs (repeatable)
        --output-dir: Directory for saving plots
        --title: Title for the plots
        --times: Specific time indices to plot
        --primary: Use primary times instead of custom
        --animation: Create animation frames
        --create-config: Generate example configuration file
        
    Notes:
        Requires at least 2 simulations for comparison. Failed file loads are
        reported and excluded from comparison. Includes example usage in help text.
    """
    parser = argparse.ArgumentParser(
        description='Compare 2T simulation results from multiple methods.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using configuration file
  python multi_compare_2T_plots.py --config comparison_config.json
  
  # Command line specification
  python multi_compare_2T_plots.py \\
      --simulation sim1.hdf5 "Method 1" \\
      --simulation sim2.hdf5 "Method 2" \\
      --simulation sim3.hdf5 "Method 3" \\
      --output-dir comparison_plots \\
      --title "Method Comparison"
  
  # Create example config file
  python multi_compare_2T_plots.py --create-config
        """
    )
    
    parser.add_argument('--config', type=str,
                        help='JSON configuration file with simulation details')
    parser.add_argument('--simulation', nargs=2, metavar=('FILE', 'LABEL'), 
                        action='append',
                        help='Simulation file and label (can be used multiple times)')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory to save output plots (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--title', type=str, default=DEFAULT_TITLE,
                        help=f'Title for the plots (default: "{DEFAULT_TITLE}")')
    parser.add_argument('--times', type=int, nargs='+',
                        help='Indices of times to plot (default: all times)')
    parser.add_argument('--primary', action='store_true',
                        help='Use primary output times instead of custom times')
    parser.add_argument('--animation', action='store_true',
                        help='Create frames for animation')
    parser.add_argument('--animation-skip', type=int, default=1,
                        help='Number of time steps to skip between animation frames (default: 1)')
    parser.add_argument('--create-config', action='store_true',
                        help='Create an example configuration file and exit')
    
    args = parser.parse_args()
    
    # Handle config file creation
    if args.create_config:
        create_example_config()
        return
    
    # Determine simulation setup method
    if args.config:
        # Load from configuration file
        config = load_config(args.config)
        if not config:
            return
        
        simulation_list = [(item['file'], item['label']) for item in config['simulations']]
        output_dir = Path(config.get('output_dir', DEFAULT_OUTPUT_DIR))
        title = config.get('title', DEFAULT_TITLE)
        use_custom = config.get('use_custom', True) and not args.primary
        
    else:
        # Use command line arguments
        if not args.simulation:
            print("Error: Must specify either --config or at least one --simulation")
            parser.print_help()
            return
        
        simulation_list = [tuple(simulation) for simulation in args.simulation]
        output_dir = Path(args.output_dir)
        title = args.title
        use_custom = not args.primary
    
    # Validate we have at least 2 simulations
    if len(simulation_list) < 2:
        print("Error: Need at least 2 simulations to compare")
        return
    
    # Set up simulation data
    simulations = setup_simulation_data(simulation_list)
    
    # Load all simulation data
    failed_loads = []
    for sim in simulations:
        if not sim.load():
            failed_loads.append(sim.label)
    
    if failed_loads:
        print(f"Failed to load the following simulations: {failed_loads}")
        # Remove failed simulations
        simulations = [sim for sim in simulations if sim.file_handle is not None]
    
    if len(simulations) < 2:
        print("Error: Need at least 2 successfully loaded simulations to compare")
        return
    
    try:
        # Create comparison plots
        plot_comparison(simulations, output_dir, title, use_custom, args.times)
        
        # Create animation frames if requested
        if args.animation:
            create_animation_frames(simulations, output_dir, title, 
                                  args.animation_skip, use_custom)
        
        print(f"All plots saved to: {output_dir}")
        
    finally:
        # Clean up: close all file handles
        for sim in simulations:
            sim.close()

if __name__ == "__main__":
    main()