# Solution snapshots for the Burgers shock case (../data/shock_snapshots.json), with the
# discretization of the shock case in run_convergence.jl, at one time before and one
# after the shock forms. The mesh is coarse on purpose, so that the differences between
# the solutions stay visible.
#
# Run:  julia --project=.. run_shock_snapshots.jl

using Trixi
using OrdinaryDiffEqVerner         # Vern9

isempty(ARGS) || error("$(basename(@__FILE__)) takes no arguments, got $(ARGS)")

const ELIXIR = joinpath(@__DIR__, "elixirs", "elixir_burgers_shock.jl")

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "setup_metadata.jl"))
# The exact solution, from the same file the elixir uses for its initial condition.
include(joinpath(@__DIR__, "elixirs", "exact_solution_burgers_shock.jl"))

const LEVEL = 4
const NVIS = 24                    # sample points per element
const N_EXACT = 2000               # sample points for the exact solution

const ES_FLUX = FluxPlusDissipation(flux_ec, DissipationLocalLaxFriedrichs())
const FD = VolumeIntegralFluxDifferencing(flux_ec)
dgsem_with(p, f) = DGSEM(polydeg = p, surface_flux = f, volume_integral = FD)

# Must match run_convergence.jl's shock case, so snapshots and EOCs describe one problem.
const SHARED = (; cfl = 0.5, ode_algorithm = Vern9(), shock_speed = 1.5,
                volume_flux = flux_ec, surface_flux = ES_FLUX)

# Shock formation time; shock_speed does not change it.
const T_CRIT = t_break
const SNAPSHOTS = [(name = "smooth", t = 0.75 * T_CRIT),
                   (name = "shock", t = 1.5 * T_CRIT)]

# The ES `name`s must match the solver names of run_convergence.jl's shock case. The EC
# runs are not plotted; they back the paper's remark on why it omits them.
const RUNS = [(name = "DGSEM P2 FD", overrides = (; solver = dgsem_with(2, ES_FLUX),
                                                  analysis_polydeg = 2)),
              (name = "DGSEM P3 FD", overrides = (; solver = dgsem_with(3, ES_FLUX),
                                                  analysis_polydeg = 3)),
              (name = "FSBP E2 FD", overrides = (; fsbp_space = "E2",
                                                 analysis_polydeg = nothing)),
              (name = "DGSEM P2 FD EC", overrides = (; solver = dgsem_with(2, flux_ec),
                                                     surface_flux = flux_ec,
                                                     analysis_polydeg = 2)),
              (name = "DGSEM P3 FD EC", overrides = (; solver = dgsem_with(3, flux_ec),
                                                     surface_flux = flux_ec,
                                                     analysis_polydeg = 3)),
              (name = "FSBP E2 FD EC", overrides = (; fsbp_space = "E2", surface_flux = flux_ec,
                                                    analysis_polydeg = nothing))]

function run_case(r, t_end)
    with_elixir(ELIXIR; initial_refinement_level = LEVEL, tspan = (0.0, t_end), SHARED...,
                r.overrides...) do m
        sol, semi = getfield(m, :sol), getfield(m, :semi)
        solver = getfield(m, :solver)
        pd = PlotData1D(sol.u[end], semi; reinterpolate = true, nvisnodes = NVIS)
        # An FSBP curve is a least-squares fit in its space and misses the nodal values.
        pn = PlotData1D(sol.u[end], semi; reinterpolate = false)
        (name = r.name, x = collect(pd.x), u = collect(pd.data[:, 1]),
         x_nodes = collect(pn.x), u_nodes = collect(pn.data[:, 1]),
         # The elixir passes `adaptive = false` to `solve`, not as a variable.
         setup = setup_metadata(ELIXIR, m, semi, solver; adaptive = false))
    end
end

function exact_curve(t)
    xs = collect(range(-1, 1; length = N_EXACT))
    us = [exact_burgers(x, t; shock_speed = SHARED.shock_speed) for x in xs]
    return (x = xs, u = us,
            shock_x = shock_position(t, -1.0; shock_speed = SHARED.shock_speed))
end

panels = map(SNAPSHOTS) do snap
    println("Snapshot $(snap.name) at t = $(snap.t)")
    solvers = map(RUNS) do r
        print("  $(r.name) ")
        out = run_case(r, snap.t)
        println("done")
        out
    end
    ex = exact_curve(snap.t)
    (name = snap.name, t = snap.t, t_over_t_crit = snap.t / T_CRIT,
     n_elements = 2^LEVEL, exact = ex, solvers = solvers)
end

out = (; level = LEVEL, shock_speed = SHARED.shock_speed, cfl = SHARED.cfl,
       t_crit = T_CRIT, panels = panels)
write_dataset("shock_snapshots", out)
