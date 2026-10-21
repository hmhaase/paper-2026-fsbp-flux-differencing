# Trixi's `initial_condition_density_wave` with amplitude 0.5 instead of 0.98, so that
# the density stays well away from zero.

function initial_condition_density_wave_safe(x, t, equations::CompressibleEulerEquations1D)
    RealT = eltype(x)
    v1 = convert(RealT, 0.1)
    rho = 1 + convert(RealT, 0.5) * sinpi(2 * (x[1] - t * v1))
    rho_v1 = rho * v1
    p = 20
    rho_e = p / (equations.gamma - 1) + 0.5f0 * rho * v1^2
    return SVector(rho, rho_v1, rho_e)
end
