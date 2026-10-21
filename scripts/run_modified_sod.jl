# Runs the modified Sod shock tube (elixirs/elixir_euler_modified_sod.jl) for each entry
# of `RUNS` and stores the final-time states in ../data/sod/modified_sod.json.
#
# Run:  julia --project=.. run_modified_sod.jl [label-substring ...]
#       (no args = all runs)

using Trixi
using Printf
using OrdinaryDiffEqSSPRK

const HERE = @__DIR__
const ELIXIR = joinpath(HERE, "elixirs", "elixir_euler_modified_sod.jl")
include(joinpath(HERE, "common.jl"))
include(joinpath(HERE, "setup_metadata.jl"))

# Errors are taken against the exact Riemann solution, not the fine-grid run, since a
# discretization can carry artifacts of its own.
include(joinpath(HERE, "elixirs", "modified_sod_states.jl"))

const N_EXACT = 2000

const DATASET = "sod/modified_sod"

const LEVEL = 6
const LEVEL_REF = 10

const EC_FLUX = flux_ranocha
const ES_FLUX = FluxPlusDissipation(flux_ranocha, DissipationMatrixWintersEtal())

# Not provably entropy stable: the surface flux of Trixi's
# examples/tree_1d_dgsem/elixir_euler_modified_sod.jl. Only the surface flux differs from
# the ES runs.
const HLLC_FLUX = flux_hllc

dgsem_with(p, sflux) = DGSEM(polydeg = p, surface_flux = sflux,
                             volume_integral = VolumeIntegralFluxDifferencing(EC_FLUX))

# P3 is the equal-cost DGSEM comparison: 4 nodes per element, as for E2.
const RUNS = [(label = "E2 (FSBP)", space = "E2", surface = "ES", level = LEVEL,
               reference = false, overrides = (;)),
              (label = "P3 (DGSEM)", space = "P3", surface = "ES", level = LEVEL,
               reference = false,
               overrides = (solver = dgsem_with(3, ES_FLUX), analysis_polydeg = 3)),
              (label = "E2 (FSBP), HLLC", space = "E2", surface = "HLLC", level = LEVEL,
               reference = false, overrides = (; surface_flux = HLLC_FLUX)),
              (label = "P3 (DGSEM), HLLC", space = "P3", surface = "HLLC", level = LEVEL,
               reference = false,
               overrides = (solver = dgsem_with(3, HLLC_FLUX), analysis_polydeg = 3)),
              (label = "P3 (DGSEM), fine grid", space = "P3", surface = "ES",
               level = LEVEL_REF, reference = true,
               overrides = (solver = dgsem_with(3, ES_FLUX), analysis_polydeg = 3))]

# Guards against an override that did not apply or a changed elixir default.
function assert_fluxes(label, solver, surface)
    sflux = solver.surface_integral.surface_flux
    vflux = solver.volume_integral.volume_flux
    vflux === EC_FLUX ||
        error("$label: expected the bare entropy-conservative volume flux $(EC_FLUX), " *
              "got $(vflux)")
    if surface == "ES"
        sflux isa FluxPlusDissipation ||
            error("$label: expected an entropy-stable FluxPlusDissipation surface flux, " *
                  "got $(sflux)")
    else
        sflux === HLLC_FLUX ||
            error("$label: expected the $(HLLC_FLUX) surface flux, got $(sflux)")
    end
    return nothing
end

# No positivity limiter is used (see the elixir), so record minimum density and pressure.
function state_minima(semi, u_ode)
    u = Trixi.wrap_array(u_ode, semi)
    _, equations, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    rho_min = Inf
    p_min = Inf
    for element in eachelement(dg, cache),
        i in Trixi.eachnode(dg, cache.elements, element)

        node = Trixi.get_node_vars(u, equations, dg, i, element)
        rho_min = min(rho_min, Trixi.density(node, equations))
        p_min = min(p_min, Trixi.pressure(node, equations))
    end
    return rho_min, p_min
end

function run_case(run)
    println("Running $(run.label) at level $(run.level)...")
    with_elixir(ELIXIR; initial_refinement_level = run.level, run.overrides...) do m
        val(s) = getfield(m, s)
        sol, semi, solver = val(:sol), val(:semi), val(:solver)
        assert_fluxes(run.label, solver, run.surface)

        pd = PlotData1D(sol.u[end], semi; solution_variables = cons2cons,
                        reinterpolate = false)
        rho_min, p_min = state_minima(semi, sol.u[end])

        # Endpoint entropy only; invariants over time are in run_sod_invariants.jl.
        S0 = Trixi.integrate(Trixi.entropy, sol.u[1], semi; normalize = false)
        S1 = Trixi.integrate(Trixi.entropy, sol.u[end], semi; normalize = false)

        exact = [[riemann_cons(riemann_exact(x, sol.t[end], WL, WR, GAMMA; x0 = X0),
                               GAMMA)[c] for x in pd.x] for c in 1:3]
        # Mean absolute nodal error, not a quadrature L1 norm.
        l1 = [sum(abs, pd.data[:, c] .- exact[c]) / length(pd.x) for c in 1:3]

        @printf("    -> %s, %d steps, t = %.3f, min rho = %.4f, min p = %.4f, dS = %.3e\n",
                sol.retcode, sol.stats.naccept, sol.t[end], rho_min, p_min, S1 - S0)
        @printf("       L1 vs exact: rho %.4e, rho*u %.4e, E %.4e\n", l1[1], l1[2], l1[3])

        return (label = run.label, space = run.space, surface = run.surface,
                level = run.level,
                reference = run.reference,
                n_elements = 2^run.level,
                ndofs = Trixi.ndofs(semi),
                retcode = string(sol.retcode),
                t_end = sol.t[end],
                nsteps = sol.stats.naccept,
                rho_min = rho_min, p_min = p_min,
                entropy_initial = S0, entropy_final = S1,
                variable_names = String.(pd.variable_names),
                x = pd.x,
                data = [pd.data[:, v] for v in 1:size(pd.data, 2)],
                exact = exact, l1_error = l1,
                element_bounds = pd.mesh_vertices_x,
                # The elixir names its integrator `ode_alg`, not `ode_algorithm`, so take
                # it from the solution.
                setup = setup_metadata(ELIXIR, m, semi, solver;
                                       ode_algorithm = string(nameof(typeof(sol.alg))),
                                       adaptive = true,
                                       integrator_options = "adaptive, dt0 = 4e-4"))
    end
end

const LABEL_ARGS = unique(lowercase.(ARGS))
unmatched = filter(a -> !any(r -> occursin(a, lowercase(r.label)), RUNS), LABEL_ARGS)
isempty(unmatched) ||
    error("no run matches $(unmatched); labels: $([r.label for r in RUNS])")
selected = isempty(ARGS) ? RUNS :
           filter(r -> any(a -> occursin(a, lowercase(r.label)), LABEL_ARGS), RUNS)

# A subset of the runs goes next to the full dataset, never over it.
dataset = length(selected) == length(RUNS) ? DATASET :
          DATASET * "_" * join(replace.(LABEL_ARGS, r"[^a-z0-9]+" => "_"), "_")

results = [run_case(r) for r in selected]

# The contact and shock positions let the plot break the exact curve at the jumps.
t_end = results[1].t_end
p_star, u_star = star_pressure_velocity(WL, WR, GAMMA)
_, S_shock = riemann_wave_speeds(WL, WR, GAMMA)
x_exact = collect(range(0.0, 1.0; length = N_EXACT))
exact_curve = (x = x_exact,
               data = [[riemann_cons(riemann_exact(x, t_end, WL, WR, GAMMA; x0 = X0),
                                     GAMMA)[c] for x in x_exact] for c in 1:3],
               discontinuities = [X0 + u_star * t_end, X0 + S_shock * t_end],
               p_star = p_star, u_star = u_star)

write_dataset(dataset,
              (label = "modified_sod", exact = exact_curve,
               description = "Modified Sod shock tube (Toro, Sec. 6.4), " *
                             "entropy-stable and HLLC interface fluxes, no shock capturing",
               t_end = t_end, runs = results))
