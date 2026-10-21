using OrdinaryDiffEqLowStorageRK
using Trixi

# Manufactured solution: flat (u = C) at t = 0, relaxing to the steady state exp(α x)
# as t → ∞.

RealT = Float64

equations = InviscidBurgersEquation1D()

const alpha_ex = 1.0
const beta_ex = 5.0    # temporal decay rate
const C_ex = 1.0

# `oftype` keeps the caller's precision.
a(t) = one(t) - exp(-oftype(t, beta_ex) * t)
b(t) = oftype(t, C_ex) * exp(-oftype(t, beta_ex) * t)
a_prime(t) = oftype(t, beta_ex) * exp(-oftype(t, beta_ex) * t)
b_prime(t) = -oftype(t, C_ex) * a_prime(t)

# u(x, t) = a(t) exp(α x) + b(t)
function initial_condition_flat_to_exp(x, t, equations::InviscidBurgersEquation1D)
    RealT = eltype(x)
    tt = convert(RealT, t)
    return SVector{1, RealT}(a(tt) * exp(convert(RealT, alpha_ex) * x[1]) + b(tt))
end

# s = u_t + u u_x for the exact solution above.
function source_terms_flat_to_exp(u, x, t, equations::InviscidBurgersEquation1D)
    RealT = eltype(x)
    tt = convert(RealT, t)
    alpha = convert(RealT, alpha_ex)
    e_ax = exp(alpha * x[1])

    u_ex = a(tt) * e_ax + b(tt)
    u_t = a_prime(tt) * e_ax + b_prime(tt)
    u_x = alpha * a(tt) * e_ax

    return SVector{1, RealT}(u_t + u_ex * u_x)
end

# u > 0 everywhere, so the left boundary is inflow and the right one outflow.
boundary_condition_inflow = BoundaryConditionDirichlet(initial_condition_flat_to_exp)

function boundary_condition_outflow(u_inner, orientation, direction, x, t,
                                    surface_flux_function,
                                    equations::InviscidBurgersEquation1D)
    return flux(u_inner, orientation, equations)
end

boundary_conditions = (x_neg = boundary_condition_inflow,
                       x_pos = boundary_condition_outflow)

initial_refinement_level = 4

# With alpha_ex = 1, "E*" = span{1, x, e^x, e^{2x}} contains both u and u^2/2.
# A String, not a Symbol: a `trixi_include` override of a Symbol would become a variable
# reference.
fsbp_space = "E*"

include(joinpath(@__DIR__, "fsbp_cell_operators.jl"))

# Flux differencing needs a symmetric two-point flux, so not the dissipative `surface_flux`.
volume_flux = flux_ec
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

coordinates_min = zero(RealT)
coordinates_max = one(RealT)

surface_flux = FluxPlusDissipation(flux_ec, DissipationLocalLaxFriedrichs())

solver = DGFSBP(fsbp_cell_operator(fsbp_space,
                                   (coordinates_max - coordinates_min) /
                                   2^initial_refinement_level; RealT = RealT),
                surface_flux = surface_flux,
                volume_integral = volume_integral)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                n_cells_max = 10_000, RealT = RealT,
                periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_flat_to_exp,
                                    solver,
                                    source_terms = source_terms_flat_to_exp,
                                    boundary_conditions = boundary_conditions)

tspan = (zero(RealT), one(RealT))
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_polydeg = nothing
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_polydeg = analysis_polydeg,
                                     extra_analysis_errors = (:l2_error_primitive,
                                                              :linf_error_primitive))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

cfl = RealT(0.8)
stepsize_callback = StepsizeCallback(cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

ode_algorithm = CarpenterKennedy2N54(williamson_condition = false)
sol = solve(ode, ode_algorithm;
            dt = one(RealT), # overwritten by stepsize_callback
            ode_default_options()..., adaptive = false, callback = callbacks);
