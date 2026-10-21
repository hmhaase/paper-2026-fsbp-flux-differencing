# Agent notes

This repository regenerates the data behind one paper;
see README.md for what it is and `julia --project=. run_all.jl --list` for which dataset backs which result.

- Environment: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`, Julia 1.13.
  `Trixi.jl/` is the Trixi.jl the project resolves to; never `Pkg.add` or `Pkg.develop` another copy.
- Run: `julia --project=. run_all.jl --list` shows the jobs; `run_all.jl <job>` runs one.
  Outputs go to `data/`, which is untracked.
- Cost: every job, `--supporting` included, finishes within about 12 minutes; the longest are the two 96-bit runs to t = 100 and `exp`.
  `run_all.jl --supporting` takes roughly an hour sequentially; independent jobs can run as separate processes.
  Each job needs up to about 1.5 GB of memory, so limit concurrent jobs to the memory too, not only the cores.
  `run_all.jl` prints each job's wall time.
- `scripts/` generators use `trixi_include` into a fresh module per run and define top-level constants, so run each one in its own process, as `run_all.jl` does.
- A solved dataset's per-solver `setup` block records how it was produced.
  Read it before you infer a parameter from the code.
