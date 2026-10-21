using OrdinaryDiffEqLowStorageRK
using Trixi

# Shock: u₀(x) = -sin(π(x - 1/8)) + shock_speed on the periodic domain [-1, 1].
# Tests entropy stability on a discontinuity, not shock resolution: there is no
# shock capturing.

RealT = Float64

equations = InviscidBurgersEquation1D()

include(joinpath(@__DIR__, "exact_solution_burgers_shock.jl"))

# Nonzero so that the shock moves across elements instead of sitting on the interface at
# x = 1/8 (refinement level >= 4), and so that u >= 1/2 stays clear of the sonic point
# u = 0.
shock_speed = RealT(1.5)

function initial_condition_burgers_shock(x, t, equations::InviscidBurgersEquation1D)
    RealT = eltype(x)
    return SVector{1, RealT}(exact_burgers(x[1], convert(RealT, t);
                                           shock_speed = shock_speed))
end

# Coarse on purpose: the E2 operator converges to the Lobatto P3 one at O(h), so on a
# fine mesh the two give nearly the same solution.
initial_refinement_level = 4

coordinates_min = -one(RealT)
coordinates_max = one(RealT)

include(joinpath(@__DIR__, "fsbp_cell_operators.jl"))

# FSBP space, named as `fsbp_cell_operator` expects it, e.g. "E2", "E*" or "T*".
fsbp_space = "E2"

# The volume flux of flux differencing must be symmetric, so all dissipation sits in the
# surface flux. Set `surface_flux = flux_ec` for the entropy-conservative variant.
volume_flux = flux_ec
surface_flux = FluxPlusDissipation(flux_ec, DissipationLocalLaxFriedrichs())
solver = DGFSBP(fsbp_cell_operator(fsbp_space,
                                   (coordinates_max - coordinates_min) /
                                   2^initial_refinement_level; RealT = RealT),
                surface_flux = surface_flux,
                volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                n_cells_max = 10_000, RealT = RealT,
                periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_burgers_shock,
                                    solver,
                                    boundary_conditions = boundary_condition_periodic)

tspan = (zero(RealT), 2 * one(RealT))   # shock forms at t = 1/π ≈ 0.318
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_polydeg = nothing
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_polydeg = analysis_polydeg,
                                     extra_analysis_errors = (:l2_error_primitive,
                                                              :linf_error_primitive))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

cfl = RealT(0.5)   # reduced: the solution is discontinuous
stepsize_callback = StepsizeCallback(cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

saveat = RealT[]
ode_algorithm = CarpenterKennedy2N54(williamson_condition = false)
sol = solve(ode, ode_algorithm;
            dt = one(RealT), # overwritten by stepsize_callback
            saveat = saveat,
            ode_default_options()..., adaptive = false, callback = callbacks);
