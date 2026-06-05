# Combined Nuclear-Matter Workflows

This folder groups the clean workflows in one place without changing their
internal logic:

- `ground_state_ab_k0/`
  Ground-state-only fit for `a`, `b`, and `K_0`.
- `quantum_tc_nc_scf/`
  Quantum finite-temperature solver for `T_c` and `n_c` using Fermi statistics
  and an SCF iteration.
- `quarkyonic_sound_speed_vs2/`
  Quarkyonic zero-temperature sound-speed workflow for `v_s^2(n/n_0)`.


## Combined notebook

Use:

- `notebooks/combined_ground_state_and_quantum_analysis.ipynb`

This notebook imports all three packages and shows:

- ground-state tables and plots
- quantum `T_c`, `n_c` tables and plots
- quarkyonic sound-speed `v_s^2` tables and plots

Combined outputs can be written into:

- `results/`

## Standalone sound-speed notebook

The quarkyonic sound-speed workflow keeps its own notebook:

- `quarkyonic_sound_speed_vs2/notebooks/quarkyonic_sound_speed_vs2_analysis.ipynb`

Its outputs are stored inside:

- `quarkyonic_sound_speed_vs2/results/`
