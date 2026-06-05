# Quantum `T_c, n_c` Solver with SCF

This folder contains a separate workflow for the **finite-temperature quantum**
critical point using Fermi statistics.

It does **not** modify the existing ground-state-only package in
`ground_state_ab_k0/`.

## Goal

Given a model and its ground-state-fitted interaction parameters `a` and `b`,
compute the quantum critical point:

- `T_c`
- `n_c`

from the conditions

- `(∂P/∂n)_T = 0`
- `(∂²P/∂n²)_T = 0`

using a self-consistent-field (SCF) numerical workflow.

## Structure

- `src/quantum_tc_nc_scf/constants.py`
  Finite-temperature solver settings and default parameter grids.
- `src/quantum_tc_nc_scf/utils/numerics.py`
  Basic numerical helpers such as Simpson integration and finite differences.
- `src/quantum_tc_nc_scf/quantum_fermi.py`
  Ideal relativistic Fermi-gas integrals at finite temperature.
- `src/quantum_tc_nc_scf/scf.py`
  SCF routines:
  - inversion from density to effective chemical potential `mu*`
  - outer SCF iteration for the critical-point equations
- `src/quantum_tc_nc_scf/solver.py`
  High-level workflow that combines:
  - the existing ground-state fit from `ground_state_ab_k0`
  - the quantum SCF critical-point solver
- `src/quantum_tc_nc_scf/reporting.py`
  Table-friendly conversion helpers.
- `notebooks/quantum_tc_nc_scf_analysis.ipynb`
  Main notebook with the mathematics, tables, and plots.
- `results/`
  CSV and PNG outputs generated from the notebook.

## Notes

- The ground-state parameters `a`, `b`, and `K_0` are imported from the
  already-working `ground_state_ab_k0` package.
- The finite-temperature part uses **Fermi statistics**, not classical
  Boltzmann expressions.
- The solution strategy is fully numerical and uses only local helper routines,
  not external root-finding libraries.

