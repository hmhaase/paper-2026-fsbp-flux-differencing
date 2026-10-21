# Conserved totals and total entropy versus time for the Euler density wave, over a long
# run on a coarse mesh.
#
# The interface flux is the bare entropy-conservative flux: a dissipative one would decay
# the entropy by construction.
#
# Run:  julia --project=.. run_density_wave_invariants.jl   (options: see README.md)

using StaticArrays

include(joinpath(@__DIR__, "invariants_common.jl"))

const LEVEL = 4
const CFL = 0.5
const N_TRAVERSALS = 5

const EC_FLUX = flux_ranocha
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
              (name = "T* EC", flux = "EC", overrides = _fsbp("T*", EC_FLUX))]

# Read from the problem, so the run length follows the elixir's domain and speed. The
# velocity is constant, so any point of the initial state gives it.
function traversal_time(semi)
    mesh, equations, _, _ = Trixi.mesh_equations_solver_cache(semi)
    u0 = semi.initial_condition(SVector(0.0), 0.0, equations)
    return mesh.tree.length_level_0 / abs(u0[2] / u0[1])
end

run_invariants(InvariantCase(slug = "density_wave_invariants",
                             elixir = joinpath(@__DIR__, "elixirs",
                                               "elixir_euler_density_wave.jl"),
                             level = LEVEL,
                             cfl = CFL,
                             runs = RUNS,
                             t_end = (semi, m) -> N_TRAVERSALS * traversal_time(semi),
                             shared = (; volume_flux = EC_FLUX),
                             # A `tend=` horizon replaces the traversal count.
                             header = isnothing(HORIZON) ? (; n_traversals = N_TRAVERSALS) :
                                      (;)))
