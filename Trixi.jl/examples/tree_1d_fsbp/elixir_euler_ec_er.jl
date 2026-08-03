using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_weak_blast_wave

volume_flux = flux_ranocha
solver = DGFSBP(LobattoLegendreCellBasisConstructor{Float64}((args...) -> 3),
                surface_flux = flux_ranocha,
                volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-2.0,)
coordinates_max = (2.0,)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 5,
                n_cells_max = 10_000, periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

analysis_polydeg = nothing
tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_polydeg = analysis_polydeg,
                                     analysis_integrals = (entropy,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.8)

# (SaveSolutionCallback is not yet supported for the ragged FSBP arrays)
callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

relaxation_solver = Trixi.RelaxationSolverNewton(max_iterations = 5,
                                                 root_tol = eps(Float64),
                                                 gamma_tol = eps(Float64))
ode_alg = Trixi.RelaxationRK44(relaxation_solver = relaxation_solver)

sol = Trixi.solve(ode, ode_alg;
                  dt = 1.0, # overwritten by the stepsize_callback
                  save_everystep = false, callback = callbacks);
