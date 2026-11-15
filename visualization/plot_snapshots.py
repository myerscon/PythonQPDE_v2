"""
Snapshot plotting utility for PDE solution visualization.

This module provides functionality for plotting simulation snapshots saved during
PDE solver execution. Supports flexible snapshot selection including individual
snapshots, lists, ranges, or all available snapshots.
"""

import argparse
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def find_snapshot_files(snapshots_dir, equation_idx=None):
    """
    Find all snapshot files in the specified directory.
    
    Args:
        snapshots_dir (str): Path to the snapshots directory.
        equation_idx (int, optional): Specific equation index to filter. 
            If None, finds all equations.
    
    Returns:
        dict: Dictionary with equation indices as keys and lists of 
            (primary_interval, filepath) tuples as values.
    """
    snapshot_files = {}
    
    # List all files in the snapshots directory
    try:
        all_files = os.listdir(snapshots_dir)
    except OSError:
        print(f"Error: Cannot access directory {snapshots_dir}")
        return snapshot_files
    
    for filename in all_files:
        # Check if it's a CSV file with the expected pattern
        if not filename.endswith('.csv'):
            continue
        
        # Check if it matches equation_*_primary_*.csv pattern
        if not filename.startswith('equation_') or '_primary_' not in filename:
            continue
        
        # Parse filename: equation_{eqn}_primary_{interval}.csv
        try:
            # Remove .csv extension and split
            name_parts = filename.replace('.csv', '').split('_')
            if len(name_parts) >= 4 and name_parts[0] == 'equation' and name_parts[2] == 'primary':
                eqn = int(name_parts[1])
                primary = int(name_parts[3])
                
                # Filter by equation if specified
                if equation_idx is not None and eqn != equation_idx:
                    continue
                
                filepath = os.path.join(snapshots_dir, filename)
                
                if eqn not in snapshot_files:
                    snapshot_files[eqn] = []
                snapshot_files[eqn].append((primary, filepath))
        except (ValueError, IndexError):
            print(f"Warning: Could not parse filename {filename}")
    
    # Sort by primary interval
    for eqn in snapshot_files:
        snapshot_files[eqn].sort(key=lambda x: x[0])
    
    return snapshot_files

def load_snapshot_data(filepath):
    """
    Load snapshot data from CSV file.
    
    Args:
        filepath (str): Path to the snapshot CSV file.
    
    Returns:
        tuple: (spatial_coordinates, solution_values) arrays.
    """
    spatial_coords = []
    solution_values = []
    
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        
        for row in reader:
            spatial_coords.append(float(row[0]))
            solution_values.append(float(row[1]))
    
    return np.array(spatial_coords), np.array(solution_values)

def parse_snapshot_selection(selection_str, available_primaries):
    """
    Parse snapshot selection string into list of primary intervals.
    
    Args:
        selection_str (str): Selection string (e.g., "1,3,5" or "1-10" or "all").
        available_primaries (list): List of available primary intervals.
    
    Returns:
        list: List of selected primary intervals.
    """
    if selection_str.lower() == 'all':
        return available_primaries
    
    selected = []
    parts = selection_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Range specification
            start, end = part.split('-')
            start, end = int(start), int(end)
            selected.extend(range(start, end + 1))
        else:
            # Single value
            selected.append(int(part))
    
    # Filter to only include available primaries
    selected = [p for p in selected if p in available_primaries]
    return sorted(list(set(selected)))  # Remove duplicates and sort

def plot_snapshots(snapshots_dir, output_dir, equation_indices=None, 
                  primary_selection='all', title_prefix="Snapshot"):
    """
    Plot selected snapshots for specified equations.
    
    Args:
        snapshots_dir (str): Directory containing snapshot CSV files.
        output_dir (str): Directory to save plots.
        equation_indices (list, optional): List of equation indices to plot.
            If None, plots all available equations.
        primary_selection (str): Selection of primary intervals to plot.
            Can be 'all', single number, comma-separated list, or range.
        title_prefix (str): Prefix for plot titles.
    
    Returns:
        None
    """
    # Find all snapshot files
    all_snapshot_files = find_snapshot_files(snapshots_dir)
    
    if not all_snapshot_files:
        print(f"No snapshot files found in {snapshots_dir}")
        return
    
    # Determine which equations to plot
    if equation_indices is None:
        equation_indices = sorted(all_snapshot_files.keys())
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Plot each equation
    for eqn in equation_indices:
        if eqn not in all_snapshot_files:
            print(f"No snapshots found for equation {eqn}")
            continue
        
        # Get available primary intervals for this equation
        available_primaries = [primary for primary, _ in all_snapshot_files[eqn]]
        
        # Parse selection
        selected_primaries = parse_snapshot_selection(primary_selection, available_primaries)
        
        if not selected_primaries:
            print(f"No valid primary intervals selected for equation {eqn}")
            continue
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot each selected snapshot
        for primary in selected_primaries:
            # Find the file for this primary interval
            filepath = None
            for p, f in all_snapshot_files[eqn]:
                if p == primary:
                    filepath = f
                    break
            
            if filepath is None:
                print(f"Warning: No file found for equation {eqn}, primary {primary}")
                continue
            
            # Load and plot data
            try:
                spatial_coords, solution_values = load_snapshot_data(filepath)
                plt.plot(spatial_coords, solution_values, 
                        label=f'Primary {primary}', 
                        linestyle='-', marker='', linewidth=2)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        # Format plot
        plt.xlabel('Spatial Coordinate')
        plt.ylabel(f'Equation {eqn} Solution')
        plt.title(f'{title_prefix} - Equation {eqn}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        output_filename = f'snapshots_equation_{eqn}.png'
        plt.savefig(output_path / output_filename, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved plot for equation {eqn}: {output_filename}")

def list_available_snapshots(snapshots_dir):
    """
    List all available snapshots for inspection.
    
    Args:
        snapshots_dir (str): Directory containing snapshot files.
    
    Returns:
        None
    """
    snapshot_files = find_snapshot_files(snapshots_dir)
    
    if not snapshot_files:
        print(f"No snapshot files found in {snapshots_dir}")
        return
    
    print(f"Available snapshots in {snapshots_dir}:")
    print("-" * 50)
    
    for eqn in sorted(snapshot_files.keys()):
        primaries = [primary for primary, _ in snapshot_files[eqn]]
        print(f"Equation {eqn}: Primary intervals {min(primaries)} to {max(primaries)} "
              f"({len(primaries)} snapshots)")
        
        # Show first few and last few if there are many
        if len(primaries) <= 10:
            print(f"  Available: {primaries}")
        else:
            print(f"  Available: {primaries[:5]} ... {primaries[-5:]} "
                  f"(showing first 5 and last 5 of {len(primaries)})")
    print("-" * 50)

def main():
    """
    Main function providing command line interface for snapshot plotting.
    
    Command line usage:
        python plot_snapshots.py snapshots_dir [options]
    
    Examples:
        # Plot all snapshots for all equations
        python plot_snapshots.py problems/heat_equation/snapshots
        
        # Plot specific snapshots
        python plot_snapshots.py problems/heat_equation/snapshots --snapshots 0,5,10
        
        # Plot range of snapshots
        python plot_snapshots.py problems/heat_equation/snapshots --snapshots 0-10
        
        # Plot specific equation only
        python plot_snapshots.py problems/heat_equation/snapshots --equations 0
        
        # List available snapshots
        python plot_snapshots.py problems/heat_equation/snapshots --list
    """
    parser = argparse.ArgumentParser(
        description='Plot PDE simulation snapshots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot all snapshots for all equations
  python plot_snapshots.py problems/heat_equation/snapshots
  
  # Plot specific snapshots
  python plot_snapshots.py problems/heat_equation/snapshots --snapshots 0,5,10
  
  # Plot range of snapshots  
  python plot_snapshots.py problems/heat_equation/snapshots --snapshots 0-10
  
  # Plot specific equation only
  python plot_snapshots.py problems/heat_equation/snapshots --equations 0
  
  # List available snapshots
  python plot_snapshots.py problems/heat_equation/snapshots --list
        """
    )
    
    parser.add_argument('snapshots_dir', type=str,
                       help='Directory containing snapshot CSV files')
    parser.add_argument('--output-dir', type=str, default='snapshot_plots',
                       help='Directory to save plots (default: snapshot_plots)')
    parser.add_argument('--equations', type=int, nargs='+',
                       help='Equation indices to plot (default: all equations)')
    parser.add_argument('--snapshots', type=str, default='all',
                       help='Snapshots to plot: "all", single number, comma-separated list, or range (e.g., "0-10")')
    parser.add_argument('--title', type=str, default='Snapshot',
                       help='Prefix for plot titles (default: "Snapshot")')
    parser.add_argument('--list', action='store_true',
                       help='List available snapshots and exit')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed information')
    
    args = parser.parse_args()
    
    # Check if snapshots directory exists
    if not os.path.exists(args.snapshots_dir):
        print(f"Error: Snapshots directory {args.snapshots_dir} does not exist")
        return
    
    # List snapshots if requested
    if args.list:
        list_available_snapshots(args.snapshots_dir)
        return
    
    # Validate snapshots directory has files
    try:
        all_files = os.listdir(args.snapshots_dir)
        csv_files = [f for f in all_files if f.endswith('.csv')]
    except OSError:
        print(f"Error: Cannot access directory {args.snapshots_dir}")
        return
    
    if not csv_files:
        print(f"Error: No CSV files found in {args.snapshots_dir}")
        return
    
    if args.verbose:
        print(f"Found {len(csv_files)} CSV files in {args.snapshots_dir}")
    
    # Create plots
    plot_snapshots(
        snapshots_dir=args.snapshots_dir,
        output_dir=args.output_dir,
        equation_indices=args.equations,
        primary_selection=args.snapshots,
        title_prefix=args.title
    )
    
    print(f"Plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()