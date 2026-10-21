# Exact solution of the 2-periodic inviscid Burgers problem with
# u₀(x) = -sinpi(x - shock_x) + shock_speed; a shock forms at t = t_break.
#
# `_exact_burgers_stationary` treats shock_speed = 0: u₀ is then anti-symmetric about
# shock_x, so the shock stays there. `exact_burgers` adds shock_speed by a Galilean boost.
# The stationary solution is u = -sinpi(ξ*), where ξ* is a root of the characteristic map
# F(ξ) = ξ - t·sinpi(ξ) - xs with xs = x - shock_x, wrapped to [-1, 1).

const shock_x = 1 / 8
const t_break  = 1 / π

# Root of F on [lo, hi]; requires F(lo) <= 0 < F(hi). A Newton step that leaves the
# bracket is replaced by bisection, which also covers F′ = 0 at ±ξs. The 60
# iterations are sized for pure bisection on a bracket of width ≤ 2.
function _solve_F(xs::T, t::T, lo::T, hi::T) where {T}
    ξ = (lo + hi) / 2
    for _ in 1:60
        F  = ξ - t * sinpi(ξ) - xs
        F == 0 && return ξ
        Fp = 1 - T(π) * t * cospi(ξ)
        if F < 0; lo = ξ; else; hi = ξ; end
        ξ_n = ξ - F / Fp
        ξ_n == ξ && return ξ
        ξ = (lo < ξ_n < hi) ? ξ_n : (lo + hi) / 2
        (ξ == lo || ξ == hi) && break
    end
    return (lo + hi) / 2
end

function _exact_burgers_stationary(xval::T, t::T) where {T <: AbstractFloat}
    xs = xval - T(shock_x)

    # The brackets below need xs within one period around the shock; u₀ is 2-periodic,
    # so wrap xs to [-1, 1) first.
    xs = mod(xs + one(T), T(2)) - one(T)

    if t == 0
        return -sinpi(xs)
    end

    if t < T(t_break)
        # F′ > 0 for t < t_break, and F(±1) = ±1 - xs bracket the unique root.
        return -sinpi(_solve_F(xs, t, -one(T), one(T)))
    end

    # For t ≥ t_break, F′ < 0 on (-ξs, ξs) and F can have three roots; the entropy
    # solution takes the outer root on each side of the shock.
    ξs = acos(one(T) / (T(π) * t)) / T(π)

    if xs < 0
        return -sinpi(_solve_F(xs, t, -one(T), -ξs))
    elseif xs > 0
        return -sinpi(_solve_F(xs, t, ξs, one(T)))
    else
        return zero(T)
    end
end

# Galilean boost: if u solves Burgers, so does u(x - c·t, t) + c, whose shock moves at
# speed c. No wrapping is needed here, as `_exact_burgers_stationary` wraps x - c·t.
function exact_burgers(xval::T, t::T; shock_speed = 0) where {T <: AbstractFloat}
    c = convert(T, shock_speed)
    return _exact_burgers_stationary(xval - c * t, t) + c
end

# Shock location at time t, wrapped into [xmin, xmin + 2); for t < t_break, the point
# of steepest gradient. `shock_x` is only the t = 0 one.
function shock_position(t, xmin = -1.0; shock_speed = 0)
    return mod(shock_x + shock_speed * t - xmin, 2) + xmin
end
