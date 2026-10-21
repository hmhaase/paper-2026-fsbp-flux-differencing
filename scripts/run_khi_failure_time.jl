# Time at which the Kelvin-Helmholtz runs fail, per operator
# (../data/khi_failure_time.json).
#
# The setup is khi_setup.jl, plus both operators with the weak-form volume integral.
#
# The stepping is driven here rather than by the elixir's `solve`, which saves only at
# `frame_dt` intervals and so does not record the time of failure.
#
# Run:  julia --project=.. -t 1 run_khi_failure_time.jl

using Trixi
using OrdinaryDiffEqSSPRK

isempty(ARGS) || error("$(basename(@__FILE__)) takes no arguments, got $(ARGS)")

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "setup_metadata.jl"))
include(joinpath(@__DIR__, "khi_setup.jl"))

const T_MAX = 6.0

const WEAK_RUNS = [(name = "P3 weak", label = "DGSEM P$POLYDEG weak",
                    overrides = (; solver = dgsem_with(ES_FLUX, WEAK))),
                   (name = "E2 weak", label = "FSBP E2 weak",
                    overrides = (; volume_integral = WEAK))]

function failure_time(r)
    print("  $(r.label): ")
    # A throwaway tspan: the elixir's own solve is not the run being measured.
    with_elixir(ELIXIR; initial_refinement_level = LEVEL, tspan = (0.0, 1.0e-8),
                r.overrides...) do m
        semi = getfield(m, :semi)
        ode = semidiscretize(semi, (0.0, T_MAX))
        integrator = init(ode, getfield(m, :ode_algorithm);
                          dt = 1.0, # overwritten by the stepsize callback
                          adaptive = false, # the stepsize callback requires this
                          ode_default_options()...,
                          callback = CallbackSet(getfield(m, :stepsize_callback)))
        # No progress, rather than a particular retcode, marks a failed run.
        while integrator.t < T_MAX
            t_before = integrator.t
            step!(integrator)
            integrator.t > t_before || break
        end
        failed = integrator.t < T_MAX
        println(failed ? "fails at t = $(integrator.t)" : "reaches T_MAX = $T_MAX",
                " after $(integrator.stats.naccept) steps")
        return (; r.name, r.label, failed, t_last = integrator.t,
                nsteps = integrator.stats.naccept,
                retcode = string(integrator.sol.retcode),
                setup = setup_metadata(ELIXIR, m, semi, getfield(m, :solver);
                                       tspan = [0.0, T_MAX], adaptive = false))
    end
end

println("KHI failure times: level $LEVEL ($(2^LEVEL) x $(2^LEVEL) elements), ",
        "T_MAX = $T_MAX, $(Threads.nthreads()) thread(s)")
runs = [failure_time(r) for r in [RUNS; WEAK_RUNS]]

write_dataset("khi_failure_time",
              (; level = LEVEL, n_elements = (2^LEVEL)^2, t_max = T_MAX,
               threads = Threads.nthreads(), runs))
