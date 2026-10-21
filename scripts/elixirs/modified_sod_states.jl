# Riemann data and exact solver shared by the modified-Sod runners. These constants must
# match the initial condition and equations of elixir_euler_modified_sod.jl.

include(joinpath(@__DIR__, "exact_solution_riemann_euler.jl"))

const GAMMA = 1.4
const WL = RiemannState(1.0, 0.75, 1.0)
const WR = RiemannState(0.125, 0.0, 0.1)
const X0 = 0.3

function boundary_fluxes(W::RiemannState, equations)
    u = SVector(riemann_cons(W, GAMMA))
    return Trixi.flux(u, 1, equations), W.u * Trixi.entropy(u, equations)
end

"""
    exact_entropy_rate(equations)

Rate at which the exact solution produces mathematical entropy at the shock; negative, and
constant while every wave is inside the domain. The closed form omits the term
`WR.u * S(WR)`, so it needs `WR.u = 0`.
"""
function exact_entropy_rate(equations)
    S_of(W) = Trixi.entropy(SVector(riemann_cons(W, GAMMA)), equations)
    iszero(WR.u) || error("exact_entropy_rate assumes WR.u = 0, but WR.u = $(WR.u)")
    _, u_star = star_pressure_velocity(WL, WR, GAMMA)
    # The right wave is a shock here (p_star > WR.p), so S_max is its speed.
    _, S_shock = riemann_wave_speeds(WL, WR, GAMMA)
    # Star-right state, sampled strictly between the contact and the shock.
    W_star = riemann_exact(X0 + (u_star + S_shock) / 2, 1.0, WL, WR, GAMMA; x0 = X0)
    return S_of(W_star) * (S_shock - u_star) - S_of(WR) * S_shock
end

"""
    assert_waves_inside(t_end, xmin, xmax)

The boundary fluxes of `WL` and `WR` and the elixir's Dirichlet condition are exact only
until a wave reaches `xmin` or `xmax`.
"""
function assert_waves_inside(t_end, xmin, xmax)
    S_min, S_max = riemann_wave_speeds(WL, WR, GAMMA)
    X0 + S_min * t_end > xmin && X0 + S_max * t_end < xmax ||
        error("a wave reaches the domain boundary by t = $t_end " *
              "(left wave at $(X0 + S_min * t_end), " *
              "right wave at $(X0 + S_max * t_end)); " *
              "the constant-boundary-state assumption fails")
    return nothing
end
