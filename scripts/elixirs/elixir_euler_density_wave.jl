# Density-wave convergence case for the 1D compressible Euler equations. Velocity and
# pressure are constant, so the exact solution needs no source term.

using OrdinaryDiffEqLowStorageRK
using Trixi

include(joinpath(@__DIR__, "initial_condition_density_wave_safe.jl"))

RealT = Float64
equations = CompressibleEulerEquations1D(RealT(1.4))

initial_condition = initial_condition_density_wave_safe

# The volume flux must be symmetric (see `VolumeIntegralFluxDifferencing`), so it carries
# no dissipation.
volume_flux = flux_ranocha
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

surface_flux = FluxPlusDissipation(flux_ranocha, DissipationMatrixWintersEtal())

initial_refinement_level = 2

coordinates_min = -one(RealT)
coordinates_max = one(RealT)

include(joinpath(@__DIR__, "fsbp_cell_operators.jl"))

# FSBP space, named as `fsbp_cell_operator` expects it, e.g. "E2", "E*" or "T*".
fsbp_space = "E2"

solver = DGFSBP(fsbp_cell_operator(fsbp_space,
                                   (coordinates_max - coordinates_min) /
                                   2^initial_refinement_level; RealT = RealT),
                surface_flux = surface_flux,
                volume_integral = volume_integral)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                n_cells_max = 30_000, RealT = RealT,
                periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_condition_periodic)

tspan = (zero(RealT), 2 * one(RealT))
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 2000
# `nothing` measures errors at the FSBP nodes, without interpolation. A DGSEM override
# must pass its own polydeg to match; the default 2 * polydeg would interpolate.
analysis_polydeg = nothing
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_polydeg = analysis_polydeg)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

cfl = RealT(0.8)
stepsize_callback = StepsizeCallback(cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

ode_algorithm = CarpenterKennedy2N54(williamson_condition = false)
sol = solve(ode, ode_algorithm;
            dt = stepsize_callback(ode), # overwritten by the stepsize_callback
            # The stepsize callback rejects adaptive stepping, so `adaptive = false` lets
            # the runners substitute an integrator that is adaptive by default.
            ode_default_options()..., adaptive = false, callback = callbacks);
