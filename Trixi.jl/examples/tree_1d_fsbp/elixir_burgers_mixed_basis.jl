using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the (inviscid) Burgers' equation

equations = InviscidBurgersEquation1D()

initial_condition = initial_condition_convergence_test

# Mix two basis families across the mesh: Lobatto-Legendre on the left half,
# a Gauss-Legendre-based GeneralSBPCellBasis (interior-only nodes) on the right.
polydeg = 3
nodes_ref, weights_ref = Trixi.gauss_nodes_weights(polydeg + 1)
D_ref = Trixi.polynomial_derivative_matrix(nodes_ref)
calc_interp = nodes_out -> Trixi.polynomial_interpolation_matrix(nodes_ref, nodes_out)

constructors = (lobatto = LobattoLegendreCellBasisConstructor{Float64}((args...) -> polydeg),
                general = ReferenceCellBasisConstructor{Float64,
                                                        Trixi.GeneralSBPCellBasis{Float64}}(D_ref,
                                                                                            nodes_ref,
                                                                                            weights_ref,
                                                                                            calc_interp,
                                                                                            (-1.0,
                                                                                             1.0)))
selector = (ctors, pos, dx) -> pos < 0.5 ? ctors.lobatto : ctors.general
constructor = MixedCellBasisConstructor(constructors, selector)

solver = DGFSBP(constructor,
                surface_flux = flux_lax_friedrichs,
                volume_integral = VolumeIntegralWeakForm())

coordinates_min = 0.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 10_000, periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_convergence_test,
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

analysis_polydeg = nothing
tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:l2_error_primitive,
                                                              :linf_error_primitive),
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
