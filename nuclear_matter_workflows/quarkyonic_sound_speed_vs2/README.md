# Quarkyonic Sound-Speed `v_s^2` Workflow

This folder contains a separate workflow for the zero-temperature quarkyonic
sound speed.

It does **not** modify:

- `ground_state_ab_k0/`
- `quantum_tc_nc_scf/`
- `sound_speed_vs2/`
- `nuclear_matter_workflows/`

## Goal

Using the already-working ground-state fit for `a`, `b`, and `K_0`, compute the
quarkyonic sound-speed curve

`v_s^2(n_B)`

at `T = 0`.

## Physics

At fixed baryon density `n_B`, the quarkyonic construction determines the
optimal configuration by minimizing the total energy density with respect to the
quark fraction. The optimization is written in terms of the lower shell
momentum `k_bu` and the nucleon Fermi momentum `k_F`.

The total energy density is built as

`epsilon_total(n_B) = epsilon_hadronic_shell(n_B, k_bu, k_F) + epsilon_quark(k_bu)`.

After the minimum-energy branch `epsilon(n_B)` is found, the pressure and sound
speed are reconstructed numerically from

`mu_B = d epsilon / d n_B`

`P = n_B mu_B - epsilon`

`v_s^2 = (dP / dn_B) / (d epsilon / dn_B)`.

This workflow evaluates the derivatives with **local finite-difference
helpers**, not library derivative routines.

## Structure

- `src/quarkyonic_sound_speed_vs2/constants.py`
  Numerical and physical settings for the quarkyonic workflow.
- `src/quarkyonic_sound_speed_vs2/utils/numerics.py`
  Local linspace, Simpson integration, finite differences, and golden-section
  minimization.
- `src/quarkyonic_sound_speed_vs2/solver.py`
  Quarkyonic energy minimization and sound-speed solver.
- `src/quarkyonic_sound_speed_vs2/reporting.py`
  Helpers for CSV-friendly output.
- `notebooks/quarkyonic_sound_speed_vs2_analysis.ipynb`
  Main notebook with equations, tables, and plots.
- `results/`
  Output CSV and PNG files created from the notebook.

## Included models

The default notebook computes the selected curves shown in the figure style:

- VDW
- RKS
- PR
- Dieterici with `alpha = 5/3`
- Clausius with `c = 4.74 fm^3`
