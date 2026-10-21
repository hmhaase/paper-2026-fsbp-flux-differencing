# Conservation and entropy over time for the modified Sod shock tube
# (../data/sod_invariants.json); the machinery is in invariants_common.jl.
#
# Kept apart from run_modified_sod.jl because the time stepping differs: here
# invariants_common.jl drives it (relaxation RK by default), not the elixir.
#
# The domain is not periodic, so the boundary inflow from `sod_boundary_rates` is
# subtracted from each total. That is exact only while the boundary states are constant,
# which `assert_waves_inside` checks.
#
# Run:  julia --project=.. run_sod_invariants.jl            (relaxation RK44)
#       julia --project=.. run_sod_invariants.jl vern9      (Vern9, for comparison)
#       All options: see README.md.

include(joinpath(@__DIR__, "invariants_common.jl"))
include(joinpath(@__DIR__, "elixirs", "modified_sod_states.jl"))

const LEVEL = 6                    # as in run_modified_sod.jl
# Trixi's relaxation integrators have no step-size control, so every run steps at a fixed
# CFL rather than adaptively as the elixir does.
const CFL = 0.5

const EC_FLUX = flux_ranocha
const ES_FLUX = FluxPlusDissipation(flux_ranocha, DissipationMatrixWintersEtal())
const FD = VolumeIntegralFluxDifferencing(EC_FLUX)
dgsem_with(p, sflux) = DGSEM(RealT = REALT, polydeg = p, surface_flux = sflux,
                             volume_integral = FD)

_dgsem(p, f) = (; solver = dgsem_with(p, f), analysis_polydeg = p)
_fsbp(space, f) = (; fsbp_space = space, surface_flux = f, analysis_polydeg = nothing)

# The equal-cost P3/E2 pair of run_modified_sod.jl, with its ES interface flux.
const RUNS = [(name = "P3 ES", flux = "ES", overrides = _dgsem(3, ES_FLUX)),
              (name = "E2 ES", flux = "ES", overrides = _fsbp("E2", ES_FLUX))]

function sod_boundary_rates(semi, equations)
    f_L, fs_L = boundary_fluxes(WL, equations)
    f_R, fs_R = boundary_fluxes(WR, equations)
    return f_L - f_R, fs_L - fs_R
end

# Must match the elixir's `tspan`, which invariants_common.jl overrides with a placeholder
# and so cannot be read back.
const T_END = 0.2

function sod_waves_inside(semi, t_end)
    mesh, _, _, _ = Trixi.mesh_equations_solver_cache(semi)
    xmin = mesh.tree.center_level_0[1] - mesh.tree.length_level_0 / 2
    assert_waves_inside(t_end, xmin, xmin + mesh.tree.length_level_0)
end

run_invariants(InvariantCase(slug = "sod_invariants",
                             elixir = joinpath(@__DIR__, "elixirs",
                                               "elixir_euler_modified_sod.jl"),
                             level = LEVEL,
                             cfl = CFL,
                             runs = RUNS,
                             t_end = (semi, m) -> T_END,
                             check_horizon = sod_waves_inside,
                             boundary_rates = sod_boundary_rates,
                             entropy_production = (semi, equations) ->
                                 exact_entropy_rate(equations)))
