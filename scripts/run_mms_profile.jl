# Samples the exact manufactured solution of the Burgers exp case for plotting
# (../data/burgers_mms_profile.json); no solve is analyzed.
#
# The profile is taken from the elixir's `initial_condition_flat_to_exp`, so the formula
# exists in one place; the elixir is run only to bring it and the parameters into scope.
#
# Run:  julia --project=.. run_mms_profile.jl

using Trixi
using StaticArrays

isempty(ARGS) || error("$(basename(@__FILE__)) takes no arguments, got $(ARGS)")

include(joinpath(@__DIR__, "common.jl"))

const ELIXIR = joinpath(@__DIR__, "elixirs", "elixir_burgers_exp_manufactured.jl")

# Multiples of the relaxation time tau = 1/beta, so the times follow the elixir's beta_ex.
const TIMES_IN_TAU = (0.0, 0.5, 1.5)
# exp(-40) is below machine epsilon, so this samples the t -> infinity limit exactly.
const STEADY_IN_TAU = 40.0
const N_POINTS = 401

function main()
    # Only the elixir's definitions are used, so the run is kept cheap.
    with_elixir(ELIXIR; initial_refinement_level = 2, tspan = (0.0, 1.0e-8)) do m
        val(s) = getfield(m, s)
        u_exact, equations = val(:initial_condition_flat_to_exp), val(:equations)
        alpha, beta, C = val(:alpha_ex), val(:beta_ex), val(:C_ex)
        xmin, xmax = val(:coordinates_min), val(:coordinates_max)
        tau = 1 / beta

        xs = collect(range(xmin, xmax; length = N_POINTS))
        sample(t) = [u_exact(SVector(x), t, equations)[1] for x in xs]

        profiles = [(t = k * tau, t_in_tau = k, u = sample(k * tau))
                    for k in TIMES_IN_TAU]
        steady = sample(STEADY_IN_TAU * tau)

        write_dataset("burgers_mms_profile",
                      (elixir = basename(ELIXIR), alpha = alpha, beta = beta, C = C,
                       tau = tau, domain = [xmin, xmax], x = xs, profiles = profiles,
                       steady = steady))
    end
end

main()
