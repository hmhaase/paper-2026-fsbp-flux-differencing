# Modified Sod shock tube for the 1D compressible Euler equations, discretized with an
# FSBP operator for the exponential space E2 = span{1, x, e^x}.
#
# Initial data, domain, boundary conditions, final time and time integrator are those of
# Trixi's examples/tree_1d_dgsem/elixir_euler_modified_sod.jl, so a DGSEM baseline only
# needs to override `solver` and `analysis_polydeg`.

using OrdinaryDiffEqSSPRK
using Trixi

RealT = Float64
equations = CompressibleEulerEquations1D(RealT(1.4))

"""
    initial_condition_modified_sod(x, t, equations::CompressibleEulerEquations1D)

Modified Sod shock tube problem. Its left rarefaction wave is sonic, so it tests whether
a numerical solution violates the entropy condition: an entropy-satisfying solution
produces a smooth rarefaction wave.
"""
function initial_condition_modified_sod(x, t, equations::CompressibleEulerEquations1D)
    RealT = eltype(x)
    if x[1] < RealT(0.3)
        return prim2cons(SVector(one(RealT), 0.75f0, one(RealT)), equations)
    else
        return prim2cons(SVector(0.125f0, zero(RealT), RealT(0.1)), equations)
    end
end
initial_condition = initial_condition_modified_sod

# The flux-differencing kernel assumes a symmetric volume flux, so the dissipation of the
# surface flux below is left out here.
volume_flux = flux_ranocha
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

# Entropy-stable surface flux, in place of the `flux_hllc` of Trixi's elixir:
# `DissipationMatrixWintersEtal` gives one only on top of an entropy-conservative flux.
surface_flux = FluxPlusDissipation(flux_ranocha, DissipationMatrixWintersEtal())

# Shared by the FSBP operator lookup and `TreeMesh`, so the operator fits the cell width.
initial_refinement_level = 6

coordinates_min = zero(RealT)
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
                initial_refinement_level = initial_refinement_level, RealT = RealT,
                n_cells_max = 30_000,
                periodicity = false)

# The Dirichlet state is valid only until a wave reaches the boundary, which `tspan` avoids.
boundary_conditions = (; x_neg = BoundaryConditionDirichlet(initial_condition),
                       x_pos = boundary_condition_do_nothing)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

tspan = (zero(RealT), RealT(0.2))
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 100)

analysis_interval = 1000
# `nothing` selects each element's own FSBP nodes; a DGSEM override must pass its polydeg.
analysis_polydeg = nothing
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_polydeg = analysis_polydeg,
                                     analysis_integrals = (entropy,))

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

# No positivity limiter: the flux-differencing volume integral does not need one here.
ode_alg = SSPRK43()

# Override to keep intermediate states; empty keeps only the initial and final state.
saveat = RealT[]

sol = solve(ode, ode_alg;
            saveat = saveat,
            dt = RealT(4e-4), adaptive = true,
            ode_default_options()..., callback = callbacks);
