# Data generators

Each `run_*.jl` solves one study and writes JSON into `../data/`.
Run a generator from this directory with a Julia environment that provides the modified Trixi.jl, e.g. `julia --project=.. -t 1 run_convergence.jl exp_1`.
Use one thread, as `run_all.jl` does: with more, threaded reductions change the last digits.
Without arguments a generator runs all of its cases.

| script | arguments | writes |
|---|---|---|
| `run_convergence.jl` | case slugs: `exp_0_5`, `exp_1`, `exp_2`, `exp_1_ec`, `exp_1_mixed`, `shock`, `euler_density_wave`, `euler_density_wave_weak` | `<slug>.json` |
| `run_shock_snapshots.jl` | – | `shock_snapshots.json`; the runs named `… EC` use the EC interface flux and are not plotted |
| `run_mms_profile.jl` | – | `burgers_mms_profile.json` |
| `run_burgers_invariants.jl` | see below | `burgers_invariants<suffix>.json` |
| `run_density_wave_invariants.jl` | see below | `density_wave_invariants<suffix>.json` |
| `run_sod_invariants.jl` | see below | `sod_invariants<suffix>.json` |
| `run_modified_sod.jl` | run-label substrings | `sod/modified_sod.json`; a subset of runs goes to `sod/modified_sod_<args>.json` |
| `run_khi_snapshot.jl` | `fd` | `khi_snapshot.json` |
| `run_khi_failure_time.jl` | – | `khi_failure_time.json` |

The three `*_invariants.jl` generators share `invariants_common.jl` and take any of:
an integrator (`relaxation`, the default, or `vern9`, `ssprk`), a precision (`f64`, the default, or `big`), `bits=N` for the BigFloat precision, `tend=T` for a shorter horizon and `runs=A,B` for a subset of runs by the first word of their name.
Each non-default choice adds a suffix to the file name, e.g. `big tend=5 bits=96` writes `density_wave_invariants_big96_t5.json` and `big bits=96 runs=E*` writes `density_wave_invariants_big96_Estar.json`.

## Cases

- `exp_*`: Burgers with the manufactured solution `u = (1 - e^{-5t}) e^{alpha x} + e^{-5t}` on [0, 1], DGSEM P2-P4 against FSBP E2 = span{1, x, e^x} and E* = span{1, x, e^x, e^{2x}}.
  `exp_1_ec` repeats `alpha = 1` with the bare entropy-conservative interface flux;
  `exp_1_mixed` uses a basis that is P3 on the left half of the domain and E2 on the right.
- `shock`: Burgers with `u0 = 3/2 - sin(pi (x - 1/8))` on [-1, 1], integrated to 75 % of the breaking time.
- `euler_density_wave`: 1D Euler density wave, including FSBP T* = span{1, x, sin 2 pi x, cos 2 pi x}.
  `euler_density_wave_weak` uses the weak-form volume integral instead of flux differencing.
- Modified Sod shock tube (Toro, Sect. 6.4, Table 6.2, Test 1) and the 2D Kelvin-Helmholtz instability, each comparing DGSEM P3 with FSBP E2 at equal node counts.

Every convergence dataset records, per solver, refinement level and conserved variable, two errors; each error field is keyed by the variable's name (`scalar` for Burgers; `rho`, `rho_v1`, `rho_e_total` for 1D Euler).
The paper reports the first variable, u for Burgers and rho for Euler.
`l2` and `linf` are taken at the solver's own nodes: the discrete norm of the operator's quadrature P and the maximum over those nodes.
`l2_exact` and `linf_exact` give the exact solution in the same norms, for relative errors.
In the 1D cases, `l2_ref` and `linf_ref` measure every solver in one common norm: the final solution is reconstructed on each element (Lagrange interpolation for DGSEM, a least-squares fit in the function space for FSBP) and the error is integrated with 51 Gauss–Lobatto nodes, without another solve.
`l2_ref_exact` and `linf_ref_exact` give the exact solution in that norm.

## Shared files

- `common.jl`: `with_elixir`, which runs an elixir in a fresh module, and `write_dataset`.
- `setup_metadata.jl`: the per-solver `setup` block every dataset records.
- `invariants_common.jl`: time stepping and bookkeeping for the `*_invariants.jl` runs.
- `khi_setup.jl`: elixir, mesh level, fluxes and run pairs of the Kelvin-Helmholtz case.
- `nodal_difference.jl`: difference of two 2D solutions at the nodes of one of them.
- `elixirs/`: the elixirs, exact solutions and tabulated FSBP operators.
  `fsbp_cell_operators.jl` builds an element's FSBP basis from the tables, looking the operator up by cell width, so it serves any domain whose cell widths are tabulated.

## Precision

Every elixir assigns `RealT = Float64` and derives its equations, domain, mesh, time span, CFL and solver from it, so `trixi_include(...; RealT = BigFloat)` runs a case at that type.
The operator tables (`fsbp_ops_*_closed.jl`) are stored in BigFloat and are exact far below Float64 round-off;
the diagnostics printed with each table entry describe the stored operator.

`trixi_include` replaces assignments anywhere in the file, including inside function bodies, so overriding `RealT` also rewrites the `RealT = eltype(x)` lines in the initial conditions.
That is harmless while the two agree, which they do once the mesh carries the same type.
