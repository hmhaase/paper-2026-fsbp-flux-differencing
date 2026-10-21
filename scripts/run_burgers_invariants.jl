# Mass and total entropy S(u) = u^2/2 versus time for the Burgers shock case. The run
# spans the shock formation time t* = 1/pi, so it covers both the smooth and the shocked
# solution.
#
# Run:  julia --project=.. run_burgers_invariants.jl   (options: see README.md)

include(joinpath(@__DIR__, "invariants_common.jl"))

const LEVEL = 4                    # as in run_shock_snapshots.jl
const T_CRIT = 1 / pi
const T_END = 1.0                  # pi t*, so most of the run is after the shock forms
const CFL = 0.5
const SHOCK_SPEED = 1.5

const EC_FLUX = flux_ec
const ES_FLUX = FluxPlusDissipation(flux_ec, DissipationLocalLaxFriedrichs())
# The volume flux stays EC: flux differencing needs a symmetric two-point flux.
const FD = VolumeIntegralFluxDifferencing(EC_FLUX)
dgsem_with(p, sflux) = DGSEM(RealT = REALT, polydeg = p, surface_flux = sflux,
                             volume_integral = FD)

_dgsem(p, f) = (; solver = dgsem_with(p, f), analysis_polydeg = p)
_fsbp(space, f) = (; fsbp_space = space, surface_flux = f, analysis_polydeg = nothing)

const RUNS = [(name = "P2 EC", flux = "EC", overrides = _dgsem(2, EC_FLUX)),
              (name = "P3 EC", flux = "EC", overrides = _dgsem(3, EC_FLUX)),
              (name = "P4 EC", flux = "EC", overrides = _dgsem(4, EC_FLUX)),
              (name = "E2 EC", flux = "EC", overrides = _fsbp("E2", EC_FLUX)),
              (name = "E* EC", flux = "EC", overrides = _fsbp("E*", EC_FLUX)),
              (name = "P2 ES", flux = "ES", overrides = _dgsem(2, ES_FLUX)),
              (name = "P3 ES", flux = "ES", overrides = _dgsem(3, ES_FLUX)),
              (name = "P4 ES", flux = "ES", overrides = _dgsem(4, ES_FLUX)),
              (name = "E2 ES", flux = "ES", overrides = _fsbp("E2", ES_FLUX)),
              (name = "E* ES", flux = "ES", overrides = _fsbp("E*", ES_FLUX))]

run_invariants(InvariantCase(slug = "burgers_invariants",
                             elixir = joinpath(@__DIR__, "elixirs",
                                               "elixir_burgers_shock.jl"),
                             level = LEVEL,
                             cfl = CFL,
                             runs = RUNS,
                             t_end = (semi, m) -> T_END,
                             shared = (; shock_speed = SHOCK_SPEED,
                                       volume_flux = EC_FLUX),
                             header = (; shock_speed = SHOCK_SPEED, t_crit = T_CRIT)))
