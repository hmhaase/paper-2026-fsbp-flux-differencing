# Density field of the Kelvin-Helmholtz instability at the final time, for a polynomial
# operator P3 and the exponential FSBP E2, plus their difference at the E2 nodes:
#
#   fd   flux differencing, matrix dissipation  -> ../data/khi_snapshot.json
#
# The setup is in khi_setup.jl. There is no positivity limiter and no shock capturing;
# run_khi_failure_time.jl measures when the runs fail.
#
# Run:  julia --project=.. run_khi_snapshot.jl [case ...]    (no argument runs every case)

using Trixi
using Printf

const HERE = @__DIR__

include(joinpath(HERE, "common.jl"))
include(joinpath(HERE, "setup_metadata.jl"))
include(joinpath(HERE, "nodal_difference.jl"))
include(joinpath(HERE, "khi_setup.jl"))

const CASES = Dict("fd" => (runs = RUNS, t_end = 3.0, dataset = "khi_snapshot"))

function run_case(r, t_end)
    print("  $(r.label): ")
    with_elixir(ELIXIR; initial_refinement_level = LEVEL, tspan = (0.0, t_end),
                r.overrides...) do m
        val(s) = getfield(m, s)
        sol, semi, solver = val(:sol), val(:semi), val(:solver)
        Symbol(sol.retcode) == :Success ||
            error("$(r.label) did not reach t = $t_end: $(sol.retcode) at t = $(sol.t[end])")

        pd = PlotData2D(sol.u[end], semi)
        variable = findfirst(==("rho"), pd.variable_names)
        isnothing(variable) && error("no density in $(pd.variable_names)")
        rho = pd.data[variable]
        @printf("%d steps, t = %.3f, rho in [%.4f, %.4f]\n",
                sol.stats.naccept, sol.t[end], minimum(rho), maximum(rho))

        record = (name = r.name, label = r.label,
                  t = sol.t[end], nsteps = sol.stats.naccept,
                  ndofs = Trixi.ndofs(semi),
                  x = collect(pd.x), y = collect(pd.y),
                  # One row per y, as PlotData2D lays its data out.
                  rho = [collect(rho[i, :]) for i in 1:size(rho, 1)],
                  setup = setup_metadata(ELIXIR, m, semi, solver;
                                         # The elixir sets it in its `solve` call,
                                         # where setup_metadata cannot read it.
                                         adaptive = false))
        # Only `record` is saved; `semi` and the final state feed the nodal difference.
        return (; record, semi, u_ode = sol.u[end])
    end
end

function run_snapshot(case, (; runs, t_end, dataset))
    println("$case:")
    results = [run_case(r, t_end) for r in runs]

    # A pair, ordered polynomial first; `nodal_difference` refuses any other source.
    source, target = results
    # Density is conservative variable 1.
    diff = nodal_difference(source, target; variable = 1)
    @printf("\n  max |%s - %s| at the %s nodes: %.4e\n", source.record.name,
            target.record.name, target.record.name, maximum(abs, diff.values))
    difference = (; diff.x, diff.y,
                  # One row per y, as the runs' own fields are stored; `values` is
                  # indexed [x, y].
                  rho = [collect(diff.values[:, j]) for j in axes(diff.values, 2)])

    write_dataset(dataset,
                  (level = LEVEL, n_elements = 4^LEVEL, t_end = t_end,
                   runs = [r.record for r in results],
                   # Evaluated at the nodes of `target`, not on the runs' plot grid.
                   difference = (; source = source.record.name,
                                 target = target.record.name, difference...)))
end

if abspath(PROGRAM_FILE) == @__FILE__
    cases = isempty(ARGS) ? sort(collect(keys(CASES))) : unique(ARGS)
    unknown = setdiff(cases, keys(CASES))
    isempty(unknown) ||
        error("unknown case(s) $unknown; known: $(sort(collect(keys(CASES))))")
    foreach(case -> run_snapshot(case, CASES[case]), cases)
end
