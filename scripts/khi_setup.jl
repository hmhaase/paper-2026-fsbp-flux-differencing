# Kelvin-Helmholtz setup shared by every script that runs this case. The two runs of a
# pair differ only in the operator.

using Trixi

const ELIXIR = joinpath(@__DIR__, "elixirs",
                        "elixir_euler_kelvin_helmholtz_instability.jl")

const LEVEL = 5
const POLYDEG = 3

# Must match the elixir's own fluxes, which the E2 run keeps.
const ES_FLUX = FluxPlusDissipation(flux_ranocha, DissipationMatrixWintersEtal())
const FD = VolumeIntegralFluxDifferencing(flux_ranocha)

const WEAK = VolumeIntegralWeakForm()

dgsem_with(surface_flux, volume_integral) = DGSEM(; polydeg = POLYDEG, surface_flux,
                                                  volume_integral)

# The elixir's solver is FSBP E2, so an E2 run overrides only what its pair changes.
const RUNS = [(name = "P3", label = "DGSEM P$POLYDEG",
               overrides = (; solver = dgsem_with(ES_FLUX, FD))),
              (name = "E2", label = "FSBP E2", overrides = (;))]
