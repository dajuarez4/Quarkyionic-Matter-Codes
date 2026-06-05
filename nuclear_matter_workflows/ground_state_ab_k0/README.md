# Ground-State `a,b,K0` Solver

This folder contains a clean, ground-state-only version of the nuclear-matter
workflow. It keeps only the pieces needed to determine:

- the interaction parameters `a` and `b`
- the incompressibility `K_0`

It does **not** include the finite-temperature critical-point search, `T_c`, or
`n_c`.

## Layout

- `src/ground_state_ab_k0/constants.py`
  Ground-state constants and default parameter grids.
- `src/ground_state_ab_k0/models.py`
  Interaction models: VDW, RKS, PR, Clausius, Dieterici.
- `src/ground_state_ab_k0/fermi_gas.py`
  Ideal Fermi-gas functions at `T = 0`.
- `src/ground_state_ab_k0/solver.py`
  Ground-state relations, `a,b` fitting, `K_0`, and family sweeps.
- `src/ground_state_ab_k0/reporting.py`
  Small helpers to convert solver results into table-friendly dictionaries.
- `src/ground_state_ab_k0/utils/numerics.py`
  Numerical helpers: Simpson integration, bisection, golden section, derivative.
- `notebooks/ground_state_ab_k0_analysis.ipynb`
  Main notebook for tables and plots.
- `results/`
  Target directory for CSV and PNG outputs created from the notebook.

## Workflow

For a given model and optional parameter (`c` or `alpha`):

1. Use the ground-state condition at `n = n0` to express `a(b)`.
2. Solve the binding-energy condition to determine `b`.
3. Recover `a`.
4. Compute `K_0 = 9 (dP/dn)|_{n0}` numerically.

## Notebook

Open:

- `ground_state_ab_k0/notebooks/ground_state_ab_k0_analysis.ipynb`

The notebook:

- computes base-model results
- computes Clausius and Dieterici sweeps over the requested ranges
- displays tables
- saves CSV files into `results/`
- produces simple summary plots for `a`, `b`, and `K_0`

