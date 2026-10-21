# Shared machinery for the invariant-versus-time studies: every conserved total and the
# total entropy, recorded at every step. A case script supplies an `InvariantCase` and
# calls `run_invariants`.
#
# The default integrator is relaxation RK (Ranocha et al. 2020), so that the fully discrete
# entropy follows the semi-discrete one.

using Trixi
using OrdinaryDiffEqVerner          # Vern9, for a no-relaxation comparison
using OrdinaryDiffEqSSPRK

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "setup_metadata.jl"))

"""
    InvariantCase(; slug, elixir, level, cfl, runs, t_end, boundary_rates,
                    entropy_production, check_horizon, shared, header)

One invariants study. `runs` are `(name, flux, overrides)` records, `overrides` being
`trixi_include` keywords; `shared` are the overrides every run gets. `t_end(semi, m)`
receives the semidiscretization and the module the elixir was included into.
`header` holds extra fields for the JSON header.
"""
Base.@kwdef struct InvariantCase
    slug::String
    elixir::String
    level::Int
    cfl::Float64
    runs::Vector
    t_end::Function
    # (semi, equations) -> (conserved rates, entropy rate): net inflow per unit time
    # through the boundaries. The default, zero, suits a periodic domain.
    boundary_rates::Function = (semi, equations) -> (nothing, 0.0)
    # (semi, equations) -> the exact solution's entropy production rate, e.g. at a shock,
    # subtracted along with the boundary term. The default, zero, suits a smooth solution.
    entropy_production::Function = (semi, equations) -> 0.0
    # (semi, t_end) -> nothing, or an error if the case is invalid up to `t_end`. Called
    # with the horizon that actually runs, a `tend=` override included.
    check_horizon::Function = (semi, t_end) -> nothing
    shared::NamedTuple = (;)
    header::NamedTuple = (;)
end

# RelaxationSolverNewton has a single RealT parameter, so gamma_min and step_scaling
# must be passed as RealT too, not left at their Float64 defaults.
function relaxation_algorithm(RealT)
    newton = Trixi.RelaxationSolverNewton(max_iterations = 5, root_tol = eps(RealT),
                                          gamma_tol = eps(RealT),
                                          gamma_min = RealT(1) / RealT(10)^13,
                                          step_scaling = one(RealT))
    return Trixi.RelaxationRK44(relaxation_solver = newton)
end

# BigFloat runs test whether the measured drift in the invariants is a round-off artifact.
const PRECISIONS = Dict("f64" => (T = Float64, suffix = ""),
                        "big" => (T = BigFloat, suffix = "_big"))

function precision_choice(args)
    for a in args
        haskey(PRECISIONS, a) && return a
    end
    return "f64"
end

# Case scripts build their DGSEM solvers themselves (an FSBP run leaves the operator to the
# elixir), so they need the working precision to match the elixir's RealT.
const REALT = PRECISIONS[precision_choice(ARGS)].T

# Optional BigFloat working precision in bits, e.g. `bits=128`. The tabulated operators
# are 256-bit literals (see `fsbp_cell_operator`), so a precision above 256 does not refine
# them.
function bits_override(args)
    for a in args
        startswith(a, "bits=") && return parse(Int, a[(length("bits=") + 1):end])
    end
    return nothing
end

const BITS = bits_override(ARGS)
isnothing(BITS) || REALT === BigFloat || error("`bits=$BITS` needs the `big` precision")
isnothing(BITS) || setprecision(BigFloat, BITS)

bits_tag() = isnothing(BITS) ? "" : string(BITS)

# Optional horizon override, e.g. `tend=5`, to keep BigFloat runs short. It changes what the
# run measures, so it is tagged in the output file name.
function horizon_override(args)
    for a in args
        startswith(a, "tend=") && return parse(Float64, a[(length("tend=") + 1):end])
    end
    return nothing
end

const HORIZON = horizon_override(ARGS)

horizon_tag() = isnothing(HORIZON) ? "" :
                "_t" * (isinteger(HORIZON) ? string(Int(HORIZON)) :
                        replace(string(HORIZON), "." => "p"))

# Optional subset of runs by the first word of their name, e.g. `runs=P4,E*`, so long
# BigFloat runs can go to separate processes. Tagged in the output file name.
function runs_override(args)
    for a in args
        startswith(a, "runs=") && return split(a[(length("runs=") + 1):end], ",")
    end
    return nothing
end

const RUN_FILTER = runs_override(ARGS)

runs_tag() = isnothing(RUN_FILTER) ? "" :
             "_" * join(replace.(RUN_FILTER, "*" => "star"), "_")

const INTEGRATORS = Dict(
    "relaxation" => (suffix = "", label = "RelaxationRK44 + RelaxationSolverNewton"),
    "vern9" => (suffix = "_vern9", label = "Vern9"),
    "ssprk" => (suffix = "_ssprk", label = "SSPRK43"))

function integrator_choice(args)
    for a in args
        haskey(INTEGRATORS, a) && return a
    end
    return "relaxation"
end

# A typo must not silently run the default case and overwrite its dataset.
function check_args(args)
    kinds = map(args) do a
        haskey(INTEGRATORS, a) ? :integrator :
        haskey(PRECISIONS, a) ? :precision :
        startswith(a, "bits=") ? :bits :
        startswith(a, "tend=") ? :tend :
        startswith(a, "runs=") ? :runs :
        error("unknown argument `$a`; options: $(join(keys(INTEGRATORS), ", ")), " *
              "$(join(keys(PRECISIONS), ", ")), bits=N, tend=T, runs=A,B")
    end
    repeated = [k for k in unique(kinds) if count(==(k), kinds) > 1]
    isempty(repeated) || error("more than one $(join(repeated, ", ")) argument in $args")
    return nothing
end

check_args(ARGS)

function stepper(semi, t_end, cfl, choice)
    # Time and step size are kept in RealT too; Float64 here would cap the accuracy of
    # high-precision runs.
    RealT = Trixi.real(semi)
    ode = semidiscretize(semi, (zero(RealT), RealT(21) / 20 * t_end))
    cb = CallbackSet(StepsizeCallback(cfl = RealT(cfl)))
    if choice == "relaxation"
        integ = Trixi.init(ode, relaxation_algorithm(RealT); dt = one(RealT), callback = cb)
        return integ, () -> Trixi.step!(integ), () -> integ.gamma
    end
    alg = choice == "vern9" ? Vern9() : SSPRK43()
    integ = init(ode, alg; dt = one(RealT), adaptive = false, callback = cb,
                 save_everystep = false)
    return integ, () -> Trixi.step!(integ), () -> NaN
end

# The elixir's own `sol = solve(...)` is dropped, since the time stepping is driven here.
# A trailing `;` wraps a statement in a :toplevel block, so look inside one as well.
function drop_solve(e)
    is_sol(x) = x isa Expr && x.head === :(=) && x.args[1] === :sol
    is_sol(e) && return :(nothing)
    if e isa Expr && e.head === :toplevel && any(is_sol, e.args)
        return Expr(:toplevel, filter(!is_sol, e.args)...)
    end
    return e
end

function run_case(case::InvariantCase, r, choice, RealT)
    # Only the semidiscretization is taken from the elixir, so the tspan is a placeholder
    # and `cfl` is not forwarded; the time stepping in `stepper` uses `case.cfl`.
    with_elixir(case.elixir; mapexpr = drop_solve, RealT = RealT,
                initial_refinement_level = case.level,
                tspan = (zero(RealT), RealT(1) / 10^8), case.shared...,
                r.overrides...) do m
        semi = getfield(m, :semi)
        t_end = isnothing(HORIZON) ? case.t_end(semi, m) : RealT(HORIZON)
        case.check_horizon(semi, t_end)
        _, equations, _, _ = Trixi.mesh_equations_solver_cache(semi)
        var_names = collect(String.(Trixi.varnames(cons2cons, equations)))
        M(u) = [Trixi.integrate((q, eq) -> q[v], u, semi; normalize = false)
                for v in eachvariable(equations)]
        E(u) = Trixi.integrate(entropy, u, semi; normalize = false)

        # A relaxation run forces gamma = 1 on the step that lands on the integrator's
        # horizon, and that unrelaxed step must not enter the record. So the horizon lies
        # beyond t_end and stepping stops at the first step at or past t_end instead.
        integrator, do_step, get_gamma = stepper(semi, t_end, case.cfl, choice)
        ts, ms, es, gammas = [integrator.t], [M(integrator.u)], [E(integrator.u)], Float64[]
        while integrator.t < t_end
            do_step()
            push!(ts, integrator.t)
            push!(ms, M(integrator.u))
            push!(es, E(integrator.u))
            push!(gammas, get_gamma())
        end
        conserved = [[m[v] for m in ms] for v in 1:length(var_names)]

        # Conservation defect: the change in each total less the boundary inflow, which is
        # rate * t because a case's boundary rates are taken to be constant in time.
        rates, entropy_rate = case.boundary_rates(semi, equations)
        rates = isnothing(rates) ? zeros(length(var_names)) : collect(rates)
        entropy_production = case.entropy_production(semi, equations)
        conserved_defect = [[c - series[1] - rates[v] * tt
                             for (tt, c) in zip(ts, series)]
                            for (v, series) in enumerate(conserved)]
        entropy_defect = [e - es[1] - (entropy_rate + entropy_production) * tt
                          for (tt, e) in zip(ts, es)]
        # The time stepping is driven here, not by the elixir's dropped solve, so its
        # settings are passed as overrides.
        setup = setup_metadata(case.elixir, m, semi, getfield(m, :solver);
                               tspan = [0.0, t_end],
                               cfl = case.cfl,
                               ode_algorithm = INTEGRATORS[choice].label,
                               adaptive = false,
                               # Must match the solver settings in relaxation_algorithm.
                               integrator_options = choice != "relaxation" ? nothing :
                                   "max_iterations=5, root_tol=gamma_tol=eps($RealT)")
        # The defects are differenced in RealT before the Float64 cast, so they keep the
        # accuracy that the cast totals lose. `precision_bits` records RealT.
        f64(x) = Float64.(x)
        record = (name = r.name, flux = r.flux, t = f64(ts),
                  variable_names = var_names,
                  conserved = map(f64, conserved), entropy = f64(es),
                  conserved_defect = map(f64, conserved_defect),
                  entropy_defect = f64(entropy_defect),
                  boundary_rates = f64(rates), entropy_rate = Float64(entropy_rate),
                  entropy_production = Float64(entropy_production),
                  precision_bits = precision(RealT),
                  gamma_min = choice == "relaxation" ? Float64(minimum(gammas)) : nothing,
                  gamma_max = choice == "relaxation" ? Float64(maximum(gammas)) : nothing,
                  setup = setup)
        return record, t_end
    end
end

function select_runs(runs)
    isnothing(RUN_FILTER) && return runs
    known = [first(split(r.name)) for r in runs]
    unknown = setdiff(RUN_FILTER, known)
    isempty(unknown) || error("runs=$(join(unknown, ",")) not among $(unique(known))")
    return [r for (r, k) in zip(runs, known) if k in RUN_FILTER]
end

function run_invariants(case::InvariantCase)
    choice = integrator_choice(ARGS)
    prec = precision_choice(ARGS)
    RealT = PRECISIONS[prec].T
    suffix = INTEGRATORS[choice].suffix * PRECISIONS[prec].suffix * bits_tag() *
             horizon_tag() * runs_tag()
    selected = select_runs(case.runs)

    results = map(selected) do r
        print("  $(r.name) ")
        record, t_end = run_case(case, r, choice, RealT)
        println(isnothing(record.gamma_min) ? "$(length(record.t)) steps" :
                "$(length(record.t)) steps, gamma in [$(record.gamma_min), $(record.gamma_max)]")
        (record, t_end)
    end
    runs = first.(results)
    # Every run solves the same problem, so a disagreement means they do not.
    t_ends = last.(results)
    all(t -> t ≈ t_ends[1], t_ends) || error("$(case.slug): runs disagree on t_end: $t_ends")

    out = merge((; level = case.level, n_elements = 2^case.level, cfl = case.cfl),
                case.header,
                (; t_end = Float64(t_ends[1]),
                 integrator = INTEGRATORS[choice].label,
                 precision_bits = precision(RealT),
                 runs = runs))
    write_dataset("$(case.slug)$(suffix)", out)
end
