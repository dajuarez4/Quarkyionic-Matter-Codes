# asymmetric_clausius_fit

This package extends the `nuclear_matter_workflows` ground-state methodology to
isospin-asymmetric Clausius nuclear matter at `T = 0`.

It uses five constraints:

- binding energy `E/A - m = -16 MeV`
- saturation density `n0 = 0.16 fm^-3`
- symmetry energy `J = 32.5 MeV`
- slope `L = 58.9 MeV`
- incompressibility `K0`

The fit factorizes cleanly:

1. Solve the symmetric Clausius problem for `(a, b, c)` at fixed `K0`
2. Solve the isovector splits `(a_pn - a_n)/2` and `(b_pn - b_n)/2` from `J` and `L`

At `y = 1/2`, the asymmetric construction reduces exactly to the symmetric Clausius model with
`a = (a_n + a_pn) / 2` and `b = (b_n + b_pn) / 2`.
