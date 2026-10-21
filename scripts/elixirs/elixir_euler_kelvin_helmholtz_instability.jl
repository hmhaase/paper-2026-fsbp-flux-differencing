using OrdinaryDiffEqSSPRK
using Trixi

RealT = Float64
gamma = RealT(1.4)
equations = CompressibleEulerEquations2D(gamma)

# Initial condition of Rueda-Ramírez & Gassner (2021), arXiv:2102.06017.
function initial_condition_kelvin_helmholtz_instability(x, t,
                                                        equations::CompressibleEulerEquations2D)
    RealT = eltype(x)
    slope = 15
    B = tanh(slope * x[2] + 7.5f0) - tanh(slope * x[2] - 7.5f0)
    rho = 0.5f0 + 0.75f0 * B
    v1 = 0.5f0 * (B - 1)
    v2 = convert(RealT, 0.1) * sinpi(2 * x[1])
    p = 1
    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_kelvin_helmholtz_instability

coordinates_min = (-one(RealT), -one(RealT))
coordinates_max = (one(RealT), one(RealT))
initial_refinement_level = 5

# The interface dissipation is needed: with `surface_flux = flux_ranocha` alone, this setup
# fails before t = 3. Without a positivity limiter or shock capturing, the run fails with a
# NaN time step near t = 3.6, so `tspan` cannot be extended much.
surface_flux = FluxPlusDissipation(flux_ranocha, DissipationMatrixWintersEtal())
volume_flux = flux_ranocha
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

include(joinpath(@__DIR__, "fsbp_cell_operators.jl"))

# "E2" = span{1, x, e^x} has 4 nodes per cell, as many as DGSEM P3; the other space names
# `fsbp_cell_operator` accepts are listed in fsbp_cell_operators.jl.
fsbp_space = "E2"

solver = DGFSBP(fsbp_cell_operator(fsbp_space,
                                   (coordinates_max[1] - coordinates_min[1]) /
                                   2^initial_refinement_level; RealT = RealT),
                surface_flux = surface_flux, volume_integral = volume_integral)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                n_cells_max = 100_000, RealT = RealT, periodicity = true)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_periodic)

tspan = (zero(RealT), 3 * one(RealT))
ode = semidiscretize(semi, tspan)

frame_dt = RealT(0.02)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

# SaveSolutionCallback does not support DGFSBP; snapshots are kept in memory via `saveat`.

cfl = RealT(1.3)
stepsize_callback = StepsizeCallback(cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

ode_algorithm = SSPRK43()
sol = solve(ode, ode_algorithm;
            dt = one(RealT), # overwritten by the stepsize callback
            saveat = tspan[1]:frame_dt:tspan[2],
            # The stepsize callback rejects adaptive stepping, which is SSPRK43's default.
            ode_default_options()..., adaptive = false, callback = callbacks);
