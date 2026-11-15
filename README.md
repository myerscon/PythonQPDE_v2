# PythonQPDE_v2

Python solver for nonlinear partial differential equations (PDEs) built around the
hierarchical Taylor-polynomial time-stepping scheme used by quantum PDE algorithms
(Kacewicz / Gaitan), with the quantum integration step emulated classically. The goal
is a classical reference implementation for a future fault-tolerant quantum computer.

PDEs are written symbolically in an input deck. SymPy turns them into finite-difference
update expressions, which are then lambdified and time-stepped by the solver.

This is the second version of [PythonQPDE](https://github.com/myerscon/PythonQPDE). Version 2
adds a semi-implicit scheme (v1 is explicit only), vectorized solvers, and multi-equation
coupling. **Status: under active development; interfaces may change.**

## Install

Developed with Python 3.9.

```bash
pip install -r requirements.txt
```

## Run

Every problem is defined by an input deck in `problems/`:

```bash
python run.py problems/QPDE_vs_Kull/inputs_QPDE_vs_Kull.py
python run.py problems/test_1T/inputs_1T_test.py --output-dir out/
```

Results are written to the deck's `directory_name` (or `--output-dir`) as an HDF5 file
(`output.hdf5`). Plotting helpers are in `visualization/`.

## Input decks

A deck is a Python file that defines two dictionaries:

- `general_param_dict`: grid, boundary and initial conditions, time stepping, solver options.
- `sympy_param_dict`: the equations (driver functions), auxiliary functions, spatial and
  temporal variables, and constants.

`problems/QPDE_vs_Kull/` holds radiation diffusion decks (two coupled equations: radiation
and matter temperature); `problems/test_1T` and `test_2T` are 1- and 2-equation tests.

## Time-stepping modes

Set `time_discretization` in `general_param_dict`. The time axis has `n` primary intervals,
each split into `n^(k-1)` secondary steps of size `hbar`, so the final time is `hbar * n^k`.

| Mode | You provide | Derived |
|---|---|---|
| `explicit` | `N_tot`, `epsilon`, `dt` (= `hbar`) | `n`, `k` |
| `implicit` | `total_time`, `epsilon`, `dt` (= `hbar`) | `n`, `k` |
| `implicit_pick_nk` | `total_time`, `n`, `k` (defaults 126 and 3) | `hbar = total_time / n^k` |

The implicit modes are semi-implicit and are conditionally stable; a small `hbar` is needed
for strongly coupled problems. `use_vectorized` selects the vectorized or list-based solver.

## Layout

```
run.py                  command-line entry point
kacewicz_functions.py   time-partition parameters and the integration routines
solvers/                simulation base class, explicit and implicit solvers
sympy_files/            symbolic equation processing and finite-difference writers
problems/               input decks
visualization/          plotting scripts for HDF5 output
debug.py                debugging helper
```
