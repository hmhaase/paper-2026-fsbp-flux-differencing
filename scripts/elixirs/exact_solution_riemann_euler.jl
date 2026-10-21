# Exact solution of the Riemann problem for the 1D Euler equations of an ideal gas,
# following E. F. Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics, Ch. 4.

struct RiemannState{T}
    rho::T
    u::T
    p::T
end

sound_speed(W::RiemannState, gamma) = sqrt(gamma * W.p / W.rho)

# Pressure function f_K and its derivative.
function _f_and_df(p, W::RiemannState, gamma)
    a = sound_speed(W, gamma)
    if p > W.p                                  # shock
        A = 2 / ((gamma + 1) * W.rho)
        B = (gamma - 1) / (gamma + 1) * W.p
        sq = sqrt(A / (B + p))
        f = (p - W.p) * sq
        df = sq * (1 - (p - W.p) / (2 * (B + p)))
    else                                        # rarefaction
        f = 2 * a / (gamma - 1) * ((p / W.p)^((gamma - 1) / (2 * gamma)) - 1)
        df = 1 / (W.rho * a) * (p / W.p)^(-(gamma + 1) / (2 * gamma))
    end
    return f, df
end

"""
    star_pressure_velocity(WL, WR, gamma; tol, max_iter)

Pressure `p_star` and velocity `u_star` in the star region between the two nonlinear
waves. Errors if the initial data generate a vacuum.
"""
function star_pressure_velocity(WL::RiemannState, WR::RiemannState, gamma;
                                tol = 1e-14, max_iter = 100)
    aL = sound_speed(WL, gamma)
    aR = sound_speed(WR, gamma)

    2 * (aL + aR) / (gamma - 1) > WR.u - WL.u ||
        error("vacuum is generated: the pressure positivity condition is violated")

    # Two-rarefaction estimate as the starting guess.
    p = ((aL + aR - (gamma - 1) / 2 * (WR.u - WL.u)) /
         (aL / WL.p^((gamma - 1) / (2 * gamma)) +
          aR / WR.p^((gamma - 1) / (2 * gamma))))^(2 * gamma / (gamma - 1))
    p = max(p, tol)

    lo, hi = tol, Inf
    converged = false
    for _ in 1:max_iter
        fL, dfL = _f_and_df(p, WL, gamma)
        fR, dfR = _f_and_df(p, WR, gamma)
        f = fL + fR + (WR.u - WL.u)
        if f > 0
            hi = p
        else
            lo = p
        end
        if abs(f) <= tol
            converged = true
            break
        end
        p_new = p - f / (dfL + dfR)
        # If Newton leaves the bracket, bisect it, or double p while it has no upper end.
        if !(lo < p_new < hi) || !isfinite(p_new)
            p_new = isfinite(hi) ? (lo + hi) / 2 : 2 * p
        end
        abs(p_new - p) <= tol * p && (p = p_new; converged = true; break)
        p = p_new
    end
    converged || error("star-region Newton iteration did not converge")

    fL, _ = _f_and_df(p, WL, gamma)
    fR, _ = _f_and_df(p, WR, gamma)
    u = (WL.u + WR.u) / 2 + (fR - fL) / 2
    return p, u
end

# Samples one side of the fan: `sgn = +1` for the left state (S <= u_star), -1 for the
# right. Left waves travel at u - a and right waves at u + a, hence `- sgn * a` below.
function _sample_side(S, W::RiemannState, gamma, p_star, u_star, sgn)
    a = sound_speed(W, gamma)
    if p_star > W.p
        # Shock.
        ratio = p_star / W.p
        gm = (gamma - 1) / (gamma + 1)
        S_shock = W.u - sgn * a * sqrt((gamma + 1) / (2 * gamma) * ratio +
                                       (gamma - 1) / (2 * gamma))
        if sgn * (S - S_shock) <= 0
            return W
        end
        rho_star = W.rho * (ratio + gm) / (gm * ratio + 1)
        return RiemannState(rho_star, u_star, p_star)
    else
        # Rarefaction.
        S_head = W.u - sgn * a
        a_star = a * (p_star / W.p)^((gamma - 1) / (2 * gamma))
        S_tail = u_star - sgn * a_star
        if sgn * (S - S_head) <= 0
            return W
        elseif sgn * (S - S_tail) >= 0
            return RiemannState(W.rho * (p_star / W.p)^(1 / gamma), u_star, p_star)
        end
        # Inside the fan.
        c = 2 / (gamma + 1) + sgn * (gamma - 1) / ((gamma + 1) * a) * (W.u - S)
        rho = W.rho * c^(2 / (gamma - 1))
        u = 2 / (gamma + 1) * (sgn * a + (gamma - 1) / 2 * W.u + S)
        p = W.p * c^(2 * gamma / (gamma - 1))
        return RiemannState(rho, u, p)
    end
end

"""
    riemann_exact(x, t, WL, WR, gamma; x0 = 0)

Primitive state `(rho, u, p)` of the exact Riemann solution at position `x` and time `t`,
for the initial discontinuity at `x0`. At `t = 0` the initial data are returned.
"""
function riemann_exact(x, t, WL::RiemannState, WR::RiemannState, gamma; x0 = 0)
    t <= 0 && return x < x0 ? WL : WR
    p_star, u_star = star_pressure_velocity(WL, WR, gamma)
    S = (x - x0) / t
    return S <= u_star ? _sample_side(S, WL, gamma, p_star, u_star, +1) :
           _sample_side(S, WR, gamma, p_star, u_star, -1)
end

"""
    riemann_wave_speeds(WL, WR, gamma)

Extreme signal speeds `(S_min, S_max)` of the solution.
"""
function riemann_wave_speeds(WL::RiemannState, WR::RiemannState, gamma)
    p_star, u_star = star_pressure_velocity(WL, WR, gamma)
    aL = sound_speed(WL, gamma)
    aR = sound_speed(WR, gamma)
    S_min = p_star > WL.p ?
            WL.u - aL * sqrt((gamma + 1) / (2 * gamma) * p_star / WL.p +
                             (gamma - 1) / (2 * gamma)) :
            WL.u - aL
    S_max = p_star > WR.p ?
            WR.u + aR * sqrt((gamma + 1) / (2 * gamma) * p_star / WR.p +
                             (gamma - 1) / (2 * gamma)) :
            WR.u + aR
    return S_min, S_max
end

function riemann_cons(W::RiemannState, gamma)
    return (W.rho, W.rho * W.u, W.p / (gamma - 1) + W.rho * W.u^2 / 2)
end
