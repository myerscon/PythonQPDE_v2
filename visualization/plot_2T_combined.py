#!/usr/bin/env python3
"""
Custom plotting script for 2T simulation with both fields on same plot.
Creates plots matching the Kull diffusion reference style.
"""

import argparse
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

# Set plot style to match reference
plt.style.use('default')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['lines.linewidth'] = 2

def plot_2T_combined(hdf5_file, output_dir='plots', time_indices=None, skip_individual=True):
    """
    Plot both radiation and matter fields from 2T simulation on same plot.
    Matches the Kull diffusion reference style with same colors per time.
    
    Args:
        hdf5_file (str): Path to HDF5 output file
        output_dir (str): Directory to save plots
        time_indices (list): Specific time indices to plot (None for all)
        skip_individual (bool): Skip individual time plots (default True)
    """
    
    with h5py.File(hdf5_file, 'r') as f:
        # Load data
        custom_fields = f['custom_fields'][:]  # Shape: (2, N_points, N_times)
        spatial_grid = f['spatial_grid'][:]
        custom_times = f['custom_output_times'][:]
        
        print(f"Data shape: {custom_fields.shape}")
        print(f"Number of custom times: {len(custom_times)}")
        print(f"Spatial grid points: {len(spatial_grid)}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine which times to plot
    if time_indices is None:
        time_indices = range(len(custom_times))
    
    # Define colors for different times (cycling through a nice palette)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Create the combined plot
    plt.figure(figsize=(10, 7))
    
    # Plot radiation and matter fields for each time with matching colors
    for i, t_idx in enumerate(time_indices):
        if t_idx >= len(custom_times):
            continue
            
        time_val = custom_times[t_idx]
        color = colors[i % len(colors)]
        
        # Format time for legend (handle both small and larger values)
        if time_val == 0:
            time_str = "t = 0"
        elif time_val < 0.01:
            time_str = f"t = {time_val:.4f}"
        else:
            time_str = f"t = {time_val:.3f}"
        
        # Plot radiation field (solid line, bold)
        plt.plot(spatial_grid, custom_fields[0, :, t_idx], 
                linestyle='-', linewidth=3, color=color,
                label=f'Tr {time_str}')
        
        # Plot matter field (dashed line)  
        plt.plot(spatial_grid, custom_fields[1, :, t_idx], 
                linestyle='--', linewidth=2, color=color,
                label=f'Tm {time_str}')
    
    # Format plot to match reference
    plt.xlabel('z(cm)')
    plt.ylabel('T(keV)')
    plt.title('QPDE 2T Diffusion Results')
    
    # Position legend in upper right
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper right')
    
    # Add grid matching reference style
    plt.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    
    # Set background to white
    plt.gca().set_facecolor('white')
    
    plt.tight_layout()
    
    # Save main plot
    output_file = output_path / 'QPDE_2T_diffusion_results.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"Main plot saved to: {output_file}")
    
    # Skip individual plots unless requested
    if not skip_individual:
        print("Creating individual time plots...")
        for t_idx in time_indices:
            if t_idx >= len(custom_times):
                continue
                
            plt.figure(figsize=(10, 6))
            time_val = custom_times[t_idx]
            
            # Plot both fields for this time
            plt.plot(spatial_grid, custom_fields[0, :, t_idx], 
                    'r-', linewidth=3, label='Te (Radiation)')
            plt.plot(spatial_grid, custom_fields[1, :, t_idx], 
                    'r--', linewidth=2, label='Tr (Matter)')
            
            plt.xlabel('z(cm)')
            plt.ylabel('T(keV)')
            plt.title(f'QPDE 2T Diffusion Results at t = {time_val:.4f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save individual plot
            output_file = output_path / f'2T_t{time_val:.4f}.png'
            plt.savefig(output_file, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close()
        
        print(f"Individual plots saved to: {output_path}")
    else:
        print("Skipped individual time plots (use --individual to enable)")

def main():
    parser = argparse.ArgumentParser(description='Plot 2T simulation with combined fields in Kull diffusion style')
    parser.add_argument('input_file', help='Path to HDF5 output file')
    parser.add_argument('--output-dir', default='2T_plots', 
                       help='Output directory for plots')
    parser.add_argument('--times', type=int, nargs='+',
                       help='Specific time indices to plot (default: all)')
    parser.add_argument('--individual', action='store_true',
                       help='Also create individual time plots')
    
    args = parser.parse_args()
    
    plot_2T_combined(args.input_file, args.output_dir, args.times, 
                    skip_individual=not args.individual)

if __name__ == '__main__':
    main()