# Reproducibility repository

Code to regenerate every dataset shown in

> H. M. Haase, A. R. Winters, J. Glaubitz.
> *Summation-by-parts operators for general function spaces: Flux differencing, entropy stability, and accuracy.*
> 2026.

This repository produces the data behind the figures and tables.
Every dataset is a self-describing JSON file, so you can plot it with any tool.

Source: <https://github.com/hmhaase/paper-2026-fsbp-flux-differencing>.
Each release is archived on Zenodo.

## How to cite

Please cite the paper above and this repository.
The repository's citation, generated from `CITATION.cff`, is shown by GitHub's "Cite this repository" button and on its Zenodo record.
Cite the Zenodo concept DOI for the repository as a whole, or the version DOI of a release for that exact code.

## Layout

```
Trixi.jl/            Trixi.jl with FSBP support (see "Modified Trixi.jl" below)
Trixi.jl-fsbp.diff   every change made to Trixi.jl, as one patch
scripts/             the data generators; each writes JSON into data/
scripts/elixirs/     elixirs, exact solutions and tabulated FSBP operators they run
run_all.jl           runs the generators that back the paper
AGENTS.md            notes for coding agents working in this repository
data/                output, not tracked
CITATION.cff, .zenodo.json   citation metadata
LICENSE, NOTICE      licences
```

## Requirements

Julia 1.13; `Manifest.toml` was resolved with Julia 1.13.2.
`Project.toml` and `Manifest.toml` pin every dependency; Trixi.jl resolves to the copy in `Trixi.jl/`.
The repository has been regenerated and checked against the paper on Linux x86_64 with Julia 1.13.2.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Run

```bash
julia --project=. run_all.jl --list
julia --project=. run_all.jl
julia --project=. run_all.jl khi shock
julia --project=. run_all.jl --supporting
```

The first command lists each job, the files it writes, and the result in the paper it supports.
With no arguments, `run_all.jl` runs every Float64 job the paper relies on: the figures and tables, and the statements in the text that rest on a computation.
`--supporting` adds the BigFloat reruns that the density-wave section cites but does not plot: runs to t = 5 with a Float64 baseline over the same horizon, and 96-bit runs of P4 and E\* to t = 100.
Each job runs in its own Julia process, on one thread.

Running every job, `--supporting` included, takes roughly an hour one after another, or about 15 minutes as separate concurrent processes (`run_all.jl <job>`) on a 16-core machine.
Each job needs up to about 1.5 GB of memory, so limit the number of concurrent jobs to the available memory as well.

You can also run a generator directly, e.g. `julia --project=. -t 1 scripts/run_convergence.jl exp_1`; `scripts/README.md` lists every generator's arguments.

Every dataset that comes from a solve records how it was produced in a `setup` block per solver:
elixir, equations, domain, time span, time integrator, CFL, fluxes, nodes per element, dimension of the function space and Trixi.jl version.

## Modified Trixi.jl

`Trixi.jl/` is upstream Trixi.jl at commit `12f0483c3` (version 0.16.25-DEV) plus the commits that add FSBP operators:
per-element cell bases, the FSBP DG solver on `TreeMesh` in 1D and 2D, analysis and time-step callbacks, relaxation Runge–Kutta support, and plotting of per-element node sets.
Changes to upstream DGSEM code are dispatch-only and leave its results unchanged.

`Trixi.jl-fsbp.diff` holds every change as one patch against that upstream commit;
in a checkout of upstream Trixi.jl at `12f0483c3`, `git apply Trixi.jl-fsbp.diff` reproduces `Trixi.jl/`.
In a git clone of this repository the changes are also available commit by commit:
the root commit holds the upstream tree unchanged, and every later commit to `Trixi.jl/` is one of ours.

```bash
git log --oneline -- Trixi.jl
```

Upstream's CI and metadata files inside `Trixi.jl/` are kept unchanged and are inert here.
See `NOTICE` for licensing.

## Results stated in the text

Some statements in the paper rest on computations that no figure or table shows:

- **Polynomial FSBP reproduces DGSEM** (introduction): `Trixi.jl/test/test_tree_1d_burgers_fsbp.jl` and `Trixi.jl/test/test_tree_2d_fsbp.jl` run the same problems with Trixi's DGSEM and with the FSBP solver on Lobatto–Legendre bases and require the solutions to agree to 1e-12; they belong to the Trixi.jl test suite, not to `run_all.jl`.
- **Odd-even convergence with an entropy-conservative interface flux** (Burgers MMS):
  `exp_1_ec` repeats `exp_1` with the bare `flux_ec` at the interfaces for every operator.
- **Entropy-conservative solutions after the shock** (Burgers):
  `shock_snapshots` also stores P2, P3 and E2 with the bare `flux_ec` at the interfaces (runs named `… EC`), which the paper does not plot.
- **Weak-form DG with T\* is exact up to round-off** (density wave):
  `euler_density_wave_weak` runs T\* and, at the same 5 nodes, DGSEM P4 with the weak-form volume integral and the entropy-stable surface flux, levels 2-8, Vern9 at CFL 0.5, to t = 2.
  Velocity and pressure are constant, so the flux is affine in rho, and T\* contains rho at every t; the weak-form volume term is then exact.
  The `FSBP T* WF` rows stay at round-off at every level, while DGSEM P4 converges.
  Flux differencing does not inherit this because `flux_ranocha` takes logarithmic means of the densities.
- **Failure time of the Kelvin-Helmholtz runs**:
  `khi_failure_time` steps P3 and E2, with flux differencing and with the weak form (same surface flux), single-threaded, from t = 0 until a step makes no progress (the step size becomes NaN) or t = 6, and records the last time reached;
  a run reaching t = 6 is recorded as not failed.
- **Higher-precision rerun of the density wave**: `--supporting`;
  the BigFloat runs at 256, 96, 128 and 160 bits and the Float64 baseline stop at t = 5, and two 96-bit runs of P4 and E\* cover the full horizon t = 100.

## Operators

The FSBP operators are not computed at run time.
They are tabulated per element width in `scripts/elixirs/fsbp_ops_*.jl` and are the canonical form of the operators used in the paper.
They were generated with FSBP.jl, which will be made public with a later paper.
Every operator has a node on each element boundary, and every table is stored in BigFloat.

## Not used by the paper

`scripts/` also contains one case from the same study that the paper does not show, the mixed-basis Burgers run (`run_convergence.jl exp_1_mixed`).
It is included for completeness.

## Licence

MIT, see `LICENSE`.
The modified Trixi.jl in `Trixi.jl/` keeps its own MIT licence; see `NOTICE`.

Generative AI (Claude) assisted in preparing this repository.
