"""
Specialized plotting script for comparing 2T simulation results.

This module provides comparison plotting functionality for two-temperature (2T)
simulations from implicit and explicit solution methods. Creates side-by-side
comparisons with consistent styling and optional animation frame generation.

# Basic usage with default paths
python visualization/compare_2T_plots.py
# (defaults to custom output times)

# Specify custom file paths
python visualization/compare_2T_plots.py --implicit path/to/implicit.hdf5 --explicit path/to/explicit.hdf5

# Plot only specific time indices
python visualization/compare_2T_plots.py --times 0 5 10

# Plot custom output times
python visualization/compare_2T_plots.py --primary

# Create animation frames
python visualization/compare_2T_plots.py --animation --animation-skip 2
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

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
mpl.rcParams['legend.fontsize'] = 12

# Default configuration - modify these variables as needed
IMPLICIT_FILE = "problems/test_2T/may15/output_c=0.01.hdf5"
EXPLICIT_FILE = "problems/test_2T/may15/output_c=0.01_explicit.hdf5"
OUTPUT_DIR = "problems/test_2T/2T_comparison_c=0.01"

# Plot labels
TITLE = "Comparison of Implicit and Explicit 2T Simulations"
X_LABEL = "X Value"
Y_LABEL = "Unitless Temperature"
IMPLICIT_LABEL = "Implicit"
EXPLICIT_LABEL = "Explicit"
RADIATION_LABEL = "Radiation (E)"
MATTER_LABEL = "Matter (T)"

# Plot colors and styles
IMPLICIT_COLOR = 'blue'
EXPLICIT_COLOR = 'red'
RADIATION_STYLE = '-'  # Solid
MATTER_STYLE = '--'    # Dashed
LINEWIDTH = 2

def load_hdf5_file(filepath):
    """ Load and validate the HDF5 file for 2T comparison plotting.
    
    Opens HDF5 file and checks for custom output data, falling back to
    primary fields if custom data is unavailable.
    
    Args:
        filepath (str): Path to the HDF5 file.
        
    Returns:
        h5py.File: Opened HDF5 file object.
        
    Notes:
        Prioritizes custom output data for comparison but gracefully handles
        files with only primary interval data available.
    """
    try:
        f = h5py.File(filepath, 'r')
        # Check for custom fields first (preferred)
        if 'custom_fields' in f and 'custom_output_times' in f:
            print(f"Found custom output data in {filepath}")
        # Fall back to primary fields if no custom fields
        elif 'fields' in f and 'spatial_grid' in f:
            print(f"No custom output data found in {filepath}, will use primary fields")
        else:
            raise KeyError("Required datasets not found in HDF5 file")
        return f
    except Exception as e:
        print(f"Error loading HDF5 file {filepath}: {e}")
        raise

def plot_custom_comparison(implicit_file, explicit_file, output_dir):
    """ Create comparison plots for custom output times.
    
    Generates comparison plots showing both radiation and matter fields from
    implicit and explicit methods on the same plot for direct comparison.
    
    Args:
        implicit_file (h5py.File): HDF5 file for implicit solution.
        explicit_file (h5py.File): HDF5 file for explicit solution.
        output_dir (Path): Directory to save output plots.
        
    Returns:
        bool: True if successful, False if falling back to primary data.
        
    Notes:
        Uses solid lines for radiation field (field 0) and dashed lines for
        matter field (field 1). Different colors distinguish solution methods.
    """
    # Check if custom time data exists in both files
    if 'custom_fields' not in implicit_file or 'custom_fields' not in explicit_file:
        print("Custom output time data not available in one or both files")
        print("Falling back to primary field data")
        return plot_primary_comparison(implicit_file, explicit_file, output_dir)
    
    # Get data
    implicit_custom_fields = implicit_file['custom_fields']
    explicit_custom_fields = explicit_file['custom_fields']
    
    implicit_spatial_grid = implicit_file['spatial_grid'][:]
    explicit_spatial_grid = explicit_file['spatial_grid'][:]
    
    implicit_custom_times = implicit_file['custom_output_times'][:]
    explicit_custom_times = explicit_file['custom_output_times'][:]
    
    # Find common times in both datasets
    # For simplicity, use the minimum number of times from both datasets
    max_times = min(implicit_custom_times.shape[0], explicit_custom_times.shape[0])
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots for each custom time
    for time_idx in range(max_times):
        plt.figure(figsize=(10, 6))
        
        # Plot implicit solution
        plt.plot(implicit_spatial_grid, implicit_custom_fields[0, :, time_idx], 
                color=IMPLICIT_COLOR, linestyle=RADIATION_STYLE, 
                linewidth=LINEWIDTH, label=f"{IMPLICIT_LABEL} {RADIATION_LABEL}")
        
        plt.plot(implicit_spatial_grid, implicit_custom_fields[1, :, time_idx], 
                color=IMPLICIT_COLOR, linestyle=MATTER_STYLE, 
                linewidth=LINEWIDTH, label=f"{IMPLICIT_LABEL} {MATTER_LABEL}")
        
        # Plot explicit solution
        plt.plot(explicit_spatial_grid, explicit_custom_fields[0, :, time_idx], 
                color=EXPLICIT_COLOR, linestyle=RADIATION_STYLE, 
                linewidth=LINEWIDTH, label=f"{EXPLICIT_LABEL} {RADIATION_LABEL}")
        
        plt.plot(explicit_spatial_grid, explicit_custom_fields[1, :, time_idx], 
                color=EXPLICIT_COLOR, linestyle=MATTER_STYLE, 
                linewidth=LINEWIDTH, label=f"{EXPLICIT_LABEL} {MATTER_LABEL}")
        
        # For the title, use the implicit time (arbitrary choice)
        implicit_time = implicit_custom_times[time_idx]
        
        # Add labels and title
        plt.xlabel(X_LABEL)
        plt.ylabel(Y_LABEL)
        plt.title(f"{TITLE} (t = {implicit_time:.6f})")
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(output_dir / f'comparison_t{time_idx}.png', bbox_inches='tight', dpi=300)
        plt.close()
        
    return True

def plot_primary_comparison(implicit_file, explicit_file, output_dir, time_indices=None):
    """ Create comparison plots for primary interval solutions.
    
    Fallback plotting method when custom output data is unavailable, using
    primary interval solutions for method comparison.
    
    Args:
        implicit_file (h5py.File): HDF5 file for implicit solution.
        explicit_file (h5py.File): HDF5 file for explicit solution.
        output_dir (Path): Directory to save output plots.
        time_indices (list, optional): Indices of times to plot. Defaults to all.
        
    Returns:
        bool: True if successful.
        
    Notes:
        Uses same styling conventions as custom comparison plots.
        Handles mismatched time step counts between solution methods.
    """
    # Get data
    implicit_fields = implicit_file['fields']
    explicit_fields = explicit_file['fields']
    
    implicit_spatial_grid = implicit_file['spatial_grid'][:]
    explicit_spatial_grid = explicit_file['spatial_grid'][:]
    
    implicit_times = implicit_file['primary_interval_times'][:]
    explicit_times = explicit_file['primary_interval_times'][:]
    
    # Determine which time indices to plot
    if time_indices is None:
        # Find the minimum number of time steps in both files
        max_times = min(implicit_times.shape[0], explicit_times.shape[0])
        time_indices = range(max_times)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots for each selected time index
    for time_idx in time_indices:
        # Skip if time index is out of range for either file
        if (time_idx >= implicit_times.shape[0] or 
            time_idx >= explicit_times.shape[0]):
            print(f"Skipping time index {time_idx} - out of range for one of the files")
            continue
        
        plt.figure(figsize=(10, 6))
        
        # Plot implicit solution
        plt.plot(implicit_spatial_grid, implicit_fields[0, :, time_idx], 
                color=IMPLICIT_COLOR, linestyle=RADIATION_STYLE, 
                linewidth=LINEWIDTH, label=f"{IMPLICIT_LABEL} {RADIATION_LABEL}")
        
        plt.plot(implicit_spatial_grid, implicit_fields[1, :, time_idx], 
                color=IMPLICIT_COLOR, linestyle=MATTER_STYLE, 
                linewidth=LINEWIDTH, label=f"{IMPLICIT_LABEL} {MATTER_LABEL}")
        
        # Plot explicit solution
        plt.plot(explicit_spatial_grid, explicit_fields[0, :, time_idx], 
                color=EXPLICIT_COLOR, linestyle=RADIATION_STYLE, 
                linewidth=LINEWIDTH, label=f"{EXPLICIT_LABEL} {RADIATION_LABEL}")
        
        plt.plot(explicit_spatial_grid, explicit_fields[1, :, time_idx], 
                color=EXPLICIT_COLOR, linestyle=MATTER_STYLE, 
                linewidth=LINEWIDTH, label=f"{EXPLICIT_LABEL} {MATTER_LABEL}")
        
        # Add labels and title
        plt.xlabel(X_LABEL)
        plt.ylabel(Y_LABEL)
        plt.title(f"{TITLE} (t = {implicit_times[time_idx]:.6f})")
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(output_dir / f'primary_comparison_t{time_idx}.png', bbox_inches='tight', dpi=300)
        plt.close()
        
    return True

def create_animation_frames(implicit_file, explicit_file, output_dir, skip=1, use_custom=True):
    """ Create frames for animation with consistent axis scaling.
    
    Generates a series of comparison plots with consistent y-axis limits
    suitable for creating animations of solution evolution.
    
    Args:
        implicit_file (h5py.File): HDF5 file for implicit solution.
        explicit_file (h5py.File): HDF5 file for explicit solution.
        output_dir (Path): Directory to save animation frames.
        skip (int, optional): Number of time steps to skip between frames. Defaults to 1.
        use_custom (bool, optional): Use custom fields if available. Defaults to True.
        
    Notes:
        Pre-computes global min/max values for consistent scaling across all frames.
        Saves frames with zero-padded numbering for easy animation creation.
        Includes ffmpeg command suggestion for video generation.
    """
    # Check which dataset to use
    if use_custom and 'custom_fields' in implicit_file and 'custom_fields' in explicit_file:
        # Get custom data
        implicit_fields = implicit_file['custom_fields']
        explicit_fields = explicit_file['custom_fields']
        implicit_times = implicit_file['custom_output_times'][:]
        explicit_times = explicit_file['custom_output_times'][:]
        print("Using custom output data for animation")
    else:
        # Get primary data
        implicit_fields = implicit_file['fields']
        explicit_fields = explicit_file['fields']
        implicit_times = implicit_file['primary_interval_times'][:]
        explicit_times = explicit_file['primary_interval_times'][:]
        print("Using primary field data for animation")
    
    implicit_spatial_grid = implicit_file['spatial_grid'][:]
    explicit_spatial_grid = explicit_file['spatial_grid'][:]
    
    # Determine number of frames
    max_times = min(implicit_times.shape[0], explicit_times.shape[0])
    time_indices = range(0, max_times, skip)
    
    # Create animation frames directory
    frames_dir = output_dir / 'animation_frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Find global min and max values for consistent scaling
    global_min = min(
        np.min(implicit_fields[0, :, :max_times]),
        np.min(implicit_fields[1, :, :max_times]),
        np.min(explicit_fields[0, :, :max_times]),
        np.min(explicit_fields[1, :, :max_times])
    )
    
    global_max = max(
        np.max(implicit_fields[0, :, :max_times]),
        np.max(implicit_fields[1, :, :max_times]),
        np.max(explicit_fields[0, :, :max_times]),
        np.max(explicit_fields[1, :, :max_times])
    )
    
    # Add 10% padding to y-axis range
    y_range = global_max - global_min
    y_min = global_min - 0.1 * y_range
    y_max = global_max + 0.1 * y_range
    
    # Create frames
    for i, time_idx in enumerate(time_indices):
        plt.figure(figsize=(10, 6))
        
        # Plot implicit solution
        plt.plot(implicit_spatial_grid, implicit_fields[0, :, time_idx], 
                color=IMPLICIT_COLOR, linestyle=RADIATION_STYLE, 
                linewidth=LINEWIDTH, label=f"{IMPLICIT_LABEL} {RADIATION_LABEL}")
        
        plt.plot(implicit_spatial_grid, implicit_fields[1, :, time_idx], 
                color=IMPLICIT_COLOR, linestyle=MATTER_STYLE, 
                linewidth=LINEWIDTH, label=f"{IMPLICIT_LABEL} {MATTER_LABEL}")
        
        # Plot explicit solution
        plt.plot(explicit_spatial_grid, explicit_fields[0, :, time_idx], 
                color=EXPLICIT_COLOR, linestyle=RADIATION_STYLE, 
                linewidth=LINEWIDTH, label=f"{EXPLICIT_LABEL} {RADIATION_LABEL}")
        
        plt.plot(explicit_spatial_grid, explicit_fields[1, :, time_idx], 
                color=EXPLICIT_COLOR, linestyle=MATTER_STYLE, 
                linewidth=LINEWIDTH, label=f"{EXPLICIT_LABEL} {MATTER_LABEL}")
        
        # Add labels and title
        plt.xlabel(X_LABEL)
        plt.ylabel(Y_LABEL)
        plt.title(f"{TITLE} (t = {implicit_times[time_idx]:.6f})")
        plt.legend()
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
    """ Main function providing command line interface for 2T comparison plotting.
    
    Parses command line arguments and generates comparison plots for two-temperature
    simulations with options for different data sources and animation frames.
    
    Command line usage:
        python compare_2T_plots.py [options]
        
    Options:
        --implicit: Path to implicit solution HDF5 file
        --explicit: Path to explicit solution HDF5 file
        --output-dir: Directory for saving plots
        --times: Specific time indices to plot
        --primary: Use primary output times instead of custom
        --animation: Create animation frames
        --animation-skip: Frame skip interval for animations
        
    Notes:
        Default file paths are configured at the top of the module for
        convenience. Animation frames include ffmpeg command suggestions.
    """
    parser = argparse.ArgumentParser(description='Compare 2T simulation results from implicit and explicit methods.')
    parser.add_argument('--implicit', type=str, default=IMPLICIT_FILE,
                        help=f'Path to implicit solution HDF5 file (default: {IMPLICIT_FILE})')
    parser.add_argument('--explicit', type=str, default=EXPLICIT_FILE,
                        help=f'Path to explicit solution HDF5 file (default: {EXPLICIT_FILE})')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help=f'Directory to save output plots (default: {OUTPUT_DIR})')
    parser.add_argument('--times', type=int, nargs='+',
                        help='Indices of times to plot (default: all times)')
    parser.add_argument('--primary', action='store_true',
                        help='Plot primary output times instead of custom times')
    parser.add_argument('--animation', action='store_true',
                        help='Create frames for animation')
    parser.add_argument('--animation-skip', type=int, default=1,
                        help='Number of time steps to skip between animation frames (default: 1)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load HDF5 files
    implicit_file = load_hdf5_file(args.implicit)
    explicit_file = load_hdf5_file(args.explicit)
    
    try:
        if args.primary:
            plot_primary_comparison(implicit_file, explicit_file, output_dir, args.times)
        else:
            # Try custom plots first, fall back to primary if needed
            if not plot_custom_comparison(implicit_file, explicit_file, output_dir):
                print("Falling back to primary field plots")
                plot_primary_comparison(implicit_file, explicit_file, output_dir, args.times)
        
        if args.animation:
            create_animation_frames(implicit_file, explicit_file, output_dir, 
                                    args.animation_skip, not args.primary)
            
    finally:
        implicit_file.close()
        explicit_file.close()
    
    print(f"Plots have been saved to: {output_dir}")

if __name__ == "__main__":
    main()