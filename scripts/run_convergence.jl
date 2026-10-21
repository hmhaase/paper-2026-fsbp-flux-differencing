# Generates the convergence data ../data/<slug>.json from the elixirs in ./elixirs/.
#
# Per level and conserved variable, two error norms are recorded: `l2`/`linf` on each
# solver's own nodes, and, in 1D only, `l2_ref`/`linf_ref` on the common reference rule
# (ANALYSIS_POLYDEG_REF), which compare across spaces. Each `*_exact` entry is the exact
# solution in the matching norm, the denominator of the relative error.
#
# Run:  julia --project=.. run_convergence.jl [slug ...]        (no args = all cases)

using Trixi
using Printf
using OrdinaryDiffEqVerner

const ELIXIR_DIR = joinpath(@__DIR__, "elixirs")
const ELIXIR = joinpath(ELIXIR_DIR, "elixir_burgers_exp_manufactured.jl")
const ELIXIR_SHOCK = joinpath(ELIXIR_DIR, "elixir_burgers_shock.jl")
const ELIXIR_EULER = joinpath(ELIXIR_DIR, "elixir_euler_density_wave.jl")

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "setup_metadata.jl"))
# `fsbp_cell_operator`, for the mixed-basis solver built below.
include(joinpath(ELIXIR_DIR, "fsbp_cell_operators.jl"))

const ANALYSIS_POLYDEG_REF = 50

const V9 = Vern9()
const FD = VolumeIntegralFluxDifferencing(flux_ec)

# `overrides` are the trixi_include kwargs shared by all levels; `level_overrides(level)`
# adds the level-dependent ones, such as a solver built here from the cell width.
struct Run
    slug::String
    name::String
    elixir::String
    overrides::NamedTuple
    levels::UnitRange{Int}
    level_overrides::Function
end

Run(slug, name, elixir, overrides, levels) = Run(slug, name, elixir, overrides, levels,
                                                 _ -> (;))

const ES_FLUX = FluxPlusDissipation(flux_ec, DissipationLocalLaxFriedrichs())
dgsem(p, surface_flux = ES_FLUX) = DGSEM(polydeg = p, surface_flux = surface_flux,
                                         volume_integral = FD)

const EULER_FD = VolumeIntegralFluxDifferencing(flux_ranocha)
const EULER_ES_FLUX = FluxPlusDissipation(flux_ranocha, DissipationMatrixWintersEtal())
dgsem_euler(p, volume_integral = EULER_FD) = DGSEM(polydeg = p, surface_flux = EULER_ES_FLUX,
                                                   volume_integral = volume_integral)

# `fsbp_space` replaces the elixir default E*; the DGSEM rows pass `solver` and ignore it.
exp_config(alpha) = (; alpha_ex = alpha, cfl = 0.5, ode_algorithm = V9, fsbp_space = "E2")

function exp_runs(slug, alpha; surface_flux = ES_FLUX)
    c = (; exp_config(alpha)..., surface_flux = surface_flux)
    sf = surface_flux
    [Run(slug, "DGSEM P2 FD", ELIXIR, (; c..., solver = dgsem(2, sf), analysis_polydeg = 2), 1:7),
     Run(slug, "DGSEM P3 FD", ELIXIR, (; c..., solver = dgsem(3, sf), analysis_polydeg = 3), 1:7),
     Run(slug, "DGSEM P4 FD", ELIXIR, (; c..., solver = dgsem(4, sf), analysis_polydeg = 4), 1:7),
     Run(slug, "FSBP E2 FD", ELIXIR, (; c..., volume_integral = FD, analysis_polydeg = nothing), 1:7),
     Run(slug, "FSBP E* FD", ELIXIR,
         (; c..., fsbp_space = "E*", volume_integral = FD, analysis_polydeg = nothing), 1:7)]
end

# A mixed space needs the split coordinate, so its solver is built here rather than in the
# elixir; `run_elixir` checks that EXP_DOMAIN is the elixir's domain.
const EXP_DOMAIN = (0.0, 1.0)

exp_cell_width(level) = (EXP_DOMAIN[2] - EXP_DOMAIN[1]) / 2^level

function mixed_solver(level)
    DGFSBP(fsbp_cell_operator("P3|E2", exp_cell_width(level);
                              split_at = (EXP_DOMAIN[1] + EXP_DOMAIN[2]) / 2),
           surface_flux = ES_FLUX, volume_integral = FD)
end

# `fsbp_space` here only feeds the metadata; the `solver` override replaces the elixir's.
function mixed_runs(slug, alpha)
    c = exp_config(alpha)
    lv = 1:7
    [Run(slug, "DGSEM P3 FD", ELIXIR, (; c..., solver = dgsem(3), analysis_polydeg = 3), lv),
     Run(slug, "FSBP E2 FD", ELIXIR,
         (; c..., volume_integral = FD, analysis_polydeg = nothing), lv),
     Run(slug, "FSBP P3|E2 FD", ELIXIR,
         (; c..., fsbp_space = "P3|E2", analysis_polydeg = nothing), lv,
         level -> (; solver = mixed_solver(level)))]
end

# Integrated to 75% of the breaking time 1/pi, while the solution is still smooth, so the
# study measures the spatial order rather than the shock.
function shock_runs(slug)
    c = (; cfl = 0.5, ode_algorithm = V9, tspan = (0.0, 0.75 / pi), shock_speed = 1.5,
         volume_flux = flux_ec, surface_flux = ES_FLUX, fsbp_space = "E2")
    lv = 2:9
    [Run(slug, "DGSEM P2 FD", ELIXIR_SHOCK, (; c..., solver = dgsem(2), analysis_polydeg = 2), lv),
     Run(slug, "DGSEM P3 FD", ELIXIR_SHOCK, (; c..., solver = dgsem(3), analysis_polydeg = 3), lv),
     Run(slug, "DGSEM P4 FD", ELIXIR_SHOCK, (; c..., solver = dgsem(4), analysis_polydeg = 4), lv),
     Run(slug, "FSBP E2 FD", ELIXIR_SHOCK, (; c..., analysis_polydeg = nothing), lv),
     Run(slug, "FSBP E* FD", ELIXIR_SHOCK,
         (; c..., fsbp_space = "E*", analysis_polydeg = nothing), lv)]
end

# Levels 2:8 on [-1, 1], i.e. operator table keys 1:7.
function euler_runs(slug)
    # Named rather than left to the elixir defaults, so the FSBP rows share their fluxes
    # with `dgsem_euler`.
    c = (; cfl = 0.5, ode_algorithm = V9, fsbp_space = "E2",
         volume_integral = EULER_FD, surface_flux = EULER_ES_FLUX)
    lv = 2:8
    [Run(slug, "DGSEM P2 FD", ELIXIR_EULER, (; c..., solver = dgsem_euler(2), analysis_polydeg = 2), lv),
     Run(slug, "DGSEM P3 FD", ELIXIR_EULER, (; c..., solver = dgsem_euler(3), analysis_polydeg = 3), lv),
     Run(slug, "DGSEM P4 FD", ELIXIR_EULER, (; c..., solver = dgsem_euler(4), analysis_polydeg = 4), lv),
     Run(slug, "FSBP E2 FD", ELIXIR_EULER, (; c..., analysis_polydeg = nothing), lv),
     Run(slug, "FSBP E* FD", ELIXIR_EULER,
         (; c..., fsbp_space = "E*", analysis_polydeg = nothing), lv),
     # T* contains the exact density wave at every t; it has 5 nodes, as P4 does.
     Run(slug, "FSBP T* FD", ELIXIR_EULER,
         (; c..., fsbp_space = "T*", analysis_polydeg = nothing), lv)]
end

# The same density wave with the weak-form volume integral. With velocity and pressure
# constant, every flux component is affine in rho, which T* contains at every t, so only
# the time integration error remains. Flux differencing loses this, since `flux_ranocha`
# takes logarithmic means.
function euler_weak_runs(slug)
    weak = VolumeIntegralWeakForm()
    c = (; cfl = 0.5, ode_algorithm = V9, volume_integral = weak,
         surface_flux = EULER_ES_FLUX)
    lv = 2:8
    [Run(slug, "DGSEM P4 WF", ELIXIR_EULER,
         (; c..., solver = dgsem_euler(4, weak), analysis_polydeg = 4), lv),
     Run(slug, "FSBP T* WF", ELIXIR_EULER,
         (; c..., fsbp_space = "T*", analysis_polydeg = nothing), lv)]
end

# `analysis_polydeg` for the native norm, on the solver's own nodes without interpolation;
# the DGSEM default (2 * polydeg) would interpolate.
native_analysis_polydeg(solver) = solver.basis isa Trixi.PerCellBasis ? nothing :
                                  Trixi.polydeg(solver)

# Returns plain data only; see `with_elixir`.
function run_elixir(r::Run, level)
    with_elixir(r.elixir; initial_refinement_level = level, r.overrides...,
                r.level_overrides(level)...) do m
        val(s) = getfield(m, s)
        semi, sol, solver = val(:semi), val(:sol), val(:solver)
        # `mixed_solver` builds its operator from EXP_DOMAIN, not from the elixir.
        r.elixir == ELIXIR &&
            (val(:coordinates_min), val(:coordinates_max)) != EXP_DOMAIN &&
            error("$(basename(ELIXIR)) is not on $EXP_DOMAIN")
        polydeg = native_analysis_polydeg(solver)
        isdefined(m, :analysis_polydeg) && val(:analysis_polydeg) != polydeg &&
            error("$(r.name): analysis_polydeg = $(val(:analysis_polydeg)) is not the " *
                  "native $(polydeg)")
        equations = Trixi.mesh_equations_solver_cache(semi)[2]
        by_variable = values_by_variable(equations)
        acb = AnalysisCallback(semi; interval = 0, analysis_polydeg = polydeg)
        l2, linf = acb(sol)
        exact = exact_norms(semi, sol, acb)
        ref_cb = ndims(semi.mesh) == 1 ?
                 AnalysisCallback(semi; interval = 0,
                                  analysis_polydeg = ANALYSIS_POLYDEG_REF) : nothing
        ref = isnothing(ref_cb) ? nothing : ref_cb(sol)
        ref_exact = isnothing(ref_cb) ? nothing : exact_norms(semi, sol, ref_cb)
        by_variable_or_nothing(norms, f) = isnothing(norms) ? nothing :
                                           by_variable(getfield(norms, f))
        (; l2 = by_variable(l2), linf = by_variable(linf),
         l2_ref = by_variable_or_nothing(ref, :l2),
         linf_ref = by_variable_or_nothing(ref, :linf),
         l2_exact = by_variable(exact.l2), linf_exact = by_variable(exact.linf),
         l2_ref_exact = by_variable_or_nothing(ref_exact, :l2),
         linf_ref_exact = by_variable_or_nothing(ref_exact, :linf),
         nsteps = sol.stats.naccept, dofs = Trixi.ndofs(semi),
         n_elements = Trixi.nelements(solver, Trixi.mesh_equations_solver_cache(semi)[4]),
         # The elixirs run here pass `adaptive = false` to `solve`, not as a variable.
         setup = setup_metadata(r.elixir, m, semi, solver; adaptive = false))
    end
end

function run_levels(r::Run)
    print("  $(r.slug)/$(r.name): ")
    rows = map(r.levels) do level
        row = run_elixir(r, level)
        print("$(row.n_elements) ")
        return row
    end
    println("done")
    return rows
end

function values_by_variable(equations)
    names = Symbol.(Trixi.varnames(cons2cons, equations))
    return v -> NamedTuple{names}(Tuple(v))
end

# With a zero numerical solution, `calc_error_norms` returns the norms of the exact
# solution, per variable and in the same discrete norms as the errors they divide.
function exact_norms(semi, sol, analysis_callback)
    a = analysis_callback.affect!
    l2, linf = Trixi.calc_error_norms(zero(sol.u[end]), sol.t[end], a.analyzer, semi,
                                      a.cache)
    return (; l2, linf)
end

function run_all(runs)
    for slug in unique(r.slug for r in runs)
        rs = filter(r -> r.slug == slug, runs)
        println("Running: $slug")
        results = map(r -> (r, run_levels(r)), rs)
        solvers = map(results) do (r, rows)
            col(f) = [getfield(row, f) for row in rows]
            function per_variable(f)
                names = keys(getfield(rows[end], f))
                return NamedTuple{names}(Tuple([getfield(row, f)[k] for row in rows]
                                               for k in names))
            end
            l2s = first(per_variable(:l2))
            last_eoc = length(l2s) > 1 ? log2(l2s[end - 1] / l2s[end]) : NaN
            @printf("    -> l2[end]=%.3e  last-EOC=%.2f\n", l2s[end], last_eoc)
            # Omitted, not null, where no reference norm exists, so asking for one fails.
            ref = isnothing(rows[end].l2_ref) ? (;) :
                  (; l2_ref = per_variable(:l2_ref), linf_ref = per_variable(:linf_ref),
                   l2_ref_exact = per_variable(:l2_ref_exact),
                   linf_ref_exact = per_variable(:linf_ref_exact))
            # `setup` records nothing level-dependent, so the last row's stands for the run.
            (name = r.name, setup = rows[end].setup,
             l2 = per_variable(:l2), linf = per_variable(:linf), ref...,
             l2_exact = per_variable(:l2_exact), linf_exact = per_variable(:linf_exact),
             nsteps = col(:nsteps), dofs = col(:dofs))
        end
        levels = first(rs).levels
        all(r.levels == levels for r in rs) ||
            error("$slug: runs disagree on the level range")
        n_elements = [row.n_elements for row in last(first(results))]
        all([row.n_elements for row in rows] == n_elements for (_, rows) in results) ||
            error("$slug: runs disagree on the element counts")
        # The reference rule is shared: differing exact norms on it mean different problems.
        unorms = [rows[end].l2_ref_exact for (_, rows) in results]
        isnothing(first(unorms)) || all(u -> all(map(≈, u, unorms[1])), unorms) ||
            error("$slug: runs disagree on the exact solution's norm: $unorms")
        write_dataset(slug, (label = slug, n_elements = n_elements, solvers = solvers))
    end
end

const ALL_RUNS = vcat(exp_runs("exp_1", 1.0),
                      exp_runs("exp_0_5", 0.5),
                      exp_runs("exp_2", 2.0),
                      exp_runs("exp_1_ec", 1.0; surface_flux = flux_ec),
                      mixed_runs("exp_1_mixed", 1.0),
                      shock_runs("shock"),
                      euler_runs("euler_density_wave"),
                      euler_weak_runs("euler_density_wave_weak"))

if abspath(PROGRAM_FILE) == @__FILE__
    slugs = unique(r.slug for r in ALL_RUNS)
    unknown = setdiff(ARGS, slugs)
    isempty(unknown) || error("unknown case(s) $(unknown); known: $(join(slugs, ", "))")
    run_all(isempty(ARGS) ? ALL_RUNS : filter(r -> r.slug in ARGS, ALL_RUNS))
end
