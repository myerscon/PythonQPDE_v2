"""
Basic plotting utilities for PDE solution visualization.

This module provides functions for plotting simulation results from HDF5 output files,
supporting both primary interval solutions and custom output times. Includes options
for plotting specific fields, time indices, and secondary interval data.
"""

import argparse
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

def load_hdf5_file(filepath):
    """ Load and validate the HDF5 file.
    
    Opens HDF5 file and verifies required datasets are present for plotting
    simulation results.
    
    Args:
        filepath (str): Path to the HDF5 file.
        
    Returns:
        h5py.File: Opened HDF5 file object.
        
    Raises:
        KeyError: If required datasets are missing from the file.
        Exception: If file cannot be opened or accessed.
    """
    try:
        f = h5py.File(filepath, 'r')
        required_datasets = ['fields', 'spatial_grid', 'primary_interval_times', 'custom_fields', 'custom_output_times']
        for dataset in required_datasets:
            if dataset not in f:
                raise KeyError(f"Required dataset '{dataset}' not found in HDF5 file")
        return f
    except Exception as e:
        print(f"Error loading HDF5 file: {e}")
        raise

def plot_primary_times(h5file, output_dir, field_indices=None, time_indices=None, verbose=False):
    """ Plot primary interval solutions.
    
    Creates line plots showing solution evolution at primary interval times
    for specified fields and time indices.
    
    Args:
        h5file (h5py.File): HDF5 file object containing simulation data.
        output_dir (Path): Directory to save plots.
        field_indices (list, optional): Indices of fields to plot. Defaults to all.
        time_indices (list, optional): Indices of times to plot. Defaults to all.
        verbose (bool, optional): Whether to print debug information. Defaults to False.
        
    Notes:
        Each field gets its own plot with multiple time curves. Saves plots as
        PNG files named 'primary_field_{index}.png'.
    """
    fields = h5file['fields']
    spatial_grid = h5file['spatial_grid'][:]
    times = h5file['primary_interval_times'][:]
    
    # Print debug information if requested
    if verbose:
        print(f"Number of primary times: {len(times)}")
        print(f"Primary times: {times}")
        print(f"Fields shape: {fields.shape}")
    
    if field_indices is None:
        field_indices = range(fields.shape[0])
    if time_indices is None:
        time_indices = range(fields.shape[2])
    
    for field_idx in field_indices:
        plt.figure(figsize=(10, 6))
        for time_idx in time_indices:
            plt.plot(spatial_grid, fields[field_idx, :, time_idx], 
                    label=f't = {times[time_idx]:.6f}',
                    linestyle='-', marker='', markersize=2)
        
        plt.xlabel('Spatial coordinate')
        plt.ylabel(f'Field {field_idx}')
        plt.title(f'Primary Solution - Field {field_idx}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_dir / f'primary_field_{field_idx}.png', bbox_inches='tight')
        plt.close()

def plot_custom_times(h5file, output_dir, field_indices=None, verbose=False):
    """ Plot solutions at custom output times.
    
    Creates line plots showing solution evolution at user-specified custom
    output times for the specified fields.
    
    Args:
        h5file (h5py.File): HDF5 file object containing simulation data.
        output_dir (Path): Directory to save plots.
        field_indices (list, optional): Indices of fields to plot. Defaults to all.
        verbose (bool, optional): Whether to print debug information. Defaults to False.
        
    Notes:
        Uses custom_fields and custom_output_times datasets from HDF5 file.
        Saves plots as PNG files named 'custom_field_{index}.png'.
    """
    custom_fields = h5file['custom_fields']
    spatial_grid = h5file['spatial_grid'][:]
    custom_times = h5file['custom_output_times'][:]
    
    # Print debug information if requested
    if verbose:
        print(f"Number of custom times: {len(custom_times)}")
        print(f"Custom times: {custom_times}")
        print(f"Custom fields shape: {custom_fields.shape}")
    
    if field_indices is None:
        field_indices = range(custom_fields.shape[0])
    
    for field_idx in field_indices:
        plt.figure(figsize=(10, 6))
        
        # Plot custom time points
        for time_idx in range(len(custom_times)):
            # Ensure we don't exceed the available data dimensions
            if time_idx < custom_fields.shape[2]:
                plt.plot(spatial_grid, custom_fields[field_idx, :, time_idx], 
                        label=f't = {custom_times[time_idx]:.6f}',
                        linestyle='-', marker='', markersize=2)
        
        plt.xlabel('Spatial coordinate')
        plt.ylabel(f'Field {field_idx}')
        plt.title(f'Custom Time Solution - Field {field_idx}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_dir / f'custom_field_{field_idx}.png', bbox_inches='tight')
        plt.close()

def plot_secondary_times(h5file, output_dir, field_indices=None, primary_indices=None, verbose=False):
    """ Plot solutions at secondary times if available.
    
    Creates detailed plots showing solution evolution within secondary intervals
    for high temporal resolution analysis.
    
    Args:
        h5file (h5py.File): HDF5 file object containing simulation data.
        output_dir (Path): Directory to save plots.
        field_indices (list, optional): Indices of fields to plot. Defaults to all.
        primary_indices (list, optional): Indices of primary intervals. Defaults to all.
        verbose (bool, optional): Whether to print debug information. Defaults to False.
        
    Notes:
        Only available if secondary data was saved during simulation.
        Creates separate plots for each field and primary interval combination.
    """
    if 'secondary_fields' not in h5file or 'secondary_times' not in h5file:
        print("Secondary field data not available in this file")
        return
    
    secondary_fields = h5file['secondary_fields']
    spatial_grid = h5file['spatial_grid'][:]
    secondary_times = h5file['secondary_times'][:]
    
    # Print debug information if requested
    if verbose:
        print(f"Secondary fields shape: {secondary_fields.shape}")
        print(f"Secondary times shape: {secondary_times.shape}")
    
    if field_indices is None:
        field_indices = range(secondary_fields.shape[0])
    if primary_indices is None:
        primary_indices = range(secondary_fields.shape[2])
    
    for field_idx in field_indices:
        for primary_idx in primary_indices:
            plt.figure(figsize=(10, 6))
            times = secondary_times[primary_idx]
            
            for time_idx in range(secondary_fields.shape[3]):
                plt.plot(spatial_grid, secondary_fields[field_idx, :, primary_idx, time_idx],
                        label=f't = {times[time_idx]:.6f}',
                        linestyle='-', marker='', markersize=2)
            
            plt.xlabel('Spatial coordinate')
            plt.ylabel(f'Field {field_idx}')
            plt.title(f'Secondary Solutions - Field {field_idx}, Primary Interval {primary_idx}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(output_dir / f'secondary_field_{field_idx}_primary_{primary_idx}.png', bbox_inches='tight')
            plt.close()

def main():
    """ Main function providing command line interface for plotting.
    
    Parses command line arguments and generates plots based on specified
    options for plot types, field selection, and output directory.
    
    Command line usage:
        python plot_solutions.py input_file.hdf5 [options]
        
    Options:
        --plot-type: Type of plots (primary, custom, secondary, all)
        --fields: Specific field indices to plot
        --times: Specific time indices for primary plots
        --output-dir: Directory for saving plots
        --verbose: Enable detailed output information
    """
    parser = argparse.ArgumentParser(description='Plot PDE solutions from HDF5 output file.')
    parser.add_argument('input_file', type=str, help='Path to the HDF5 output file')
    parser.add_argument('--output-dir', type=str, default='plots',
                        help='Directory to save plots (default: plots)')
    parser.add_argument('--plot-type', type=str, choices=['primary', 'custom', 'secondary', 'all'],
                        default='all', help='Type of plots to generate')
    parser.add_argument('--fields', type=int, nargs='+',
                        help='Indices of fields to plot (default: all fields)')
    parser.add_argument('--times', type=int, nargs='+',
                        help='Indices of times to plot (default: all times)')
    parser.add_argument('--primary-intervals', type=int, nargs='+',
                        help='Indices of primary intervals for secondary plots (default: all intervals)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed information about the data')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load HDF5 file
    h5file = load_hdf5_file(args.input_file)
    
    try:
        if args.plot_type in ['primary', 'all']:
            plot_primary_times(h5file, output_dir, args.fields, args.times, args.verbose)
        
        if args.plot_type in ['custom', 'all']:
            plot_custom_times(h5file, output_dir, args.fields, args.verbose)
        
        if args.plot_type in ['secondary', 'all']:
            plot_secondary_times(h5file, output_dir, args.fields, args.primary_intervals, args.verbose)
            
    finally:
        h5file.close()
    
    print(f"Plots have been saved to: {output_dir}")

if __name__ == "__main__":
    main()
