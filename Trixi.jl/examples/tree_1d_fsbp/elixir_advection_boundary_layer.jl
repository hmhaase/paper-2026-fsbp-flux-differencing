using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

equations = LinearScalarAdvectionEquation1D(1.0)

# Full time-dependent exact solution, from characteristics of u_t + u_x = 2u with
# u(x,0)=1 and inflow u(0,t)=1:
#     u(x,t) = exp(2*min(x,t))     (= e^{2t} ahead of the front x>t, e^{2x} behind it).
# This one function serves three roles:
#   * IC at t=0           -> exp(0) = 1            (flat start, since x>=0)
#   * left inflow BC      -> exp(2*min(0,t)) = 1   (correct inflow u(0,t)=1)
#   * exact reference     -> true solution at every t (AnalysisCallback error norms).
# For t >= pi the whole domain [0,pi] has reached the steady state e^{2x}.
function initial_condition_boundary_layer(x, t,
                                          equations::LinearScalarAdvectionEquation1D)
    return SVector(exp(2 * min(x[1], t)))
end
initial_condition = initial_condition_boundary_layer

# Source term s(u) = 2u
source_terms_2u(u, x, t, equations::LinearScalarAdvectionEquation1D) = SVector(2 * u[1])

boundary_condition_inflow = BoundaryConditionDirichlet(initial_condition_boundary_layer)

function boundary_condition_outflow(u_inner, orientation, direction, x, t,
                                    surface_flux_function,
                                    equations::LinearScalarAdvectionEquation1D)
    return flux(u_inner, orientation, equations)
end

boundary_conditions = (x_neg = boundary_condition_inflow,
                       x_pos = boundary_condition_outflow)

polydeg = 2
nodes_ref, weights_ref = Trixi.gauss_nodes_weights(polydeg + 1)
D_ref = Trixi.polynomial_derivative_matrix(nodes_ref)
calc_interp = nodes_out -> Trixi.polynomial_interpolation_matrix(nodes_ref, nodes_out)
constructor = ReferenceCellBasisConstructor{Float64, Trixi.GeneralSBPCellBasis{Float64}}(D_ref,
                                                                                         nodes_ref,
                                                                                         weights_ref,
                                                                                         calc_interp,
                                                                                         (-1.0,
                                                                                          1.0))

volume_integral = VolumeIntegralWeakForm()
solver = DGFSBP(constructor,
                surface_flux = flux_godunov,  # dedicated upwind flux for advection
                volume_integral = volume_integral)

coordinates_min = 0.0
coordinates_max = Float64(pi)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 10_000,
                periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_2u,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

analysis_polydeg = nothing
tspan = (0.0, 3.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_polydeg = analysis_polydeg)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.8)

# (SaveSolutionCallback is not yet supported for the ragged FSBP arrays)
callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
