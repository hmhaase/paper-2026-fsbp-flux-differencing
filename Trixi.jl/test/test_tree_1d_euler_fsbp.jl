@testsnippet TreeMesh1DEulerFSBP begin
    EXAMPLES_DIR = joinpath(examples_dir(), "tree_1d_fsbp")
end

@testitem "TreeMesh1D Euler FSBP: elixir_euler_ec.jl" setup=[
    Setup,
    TreeMesh1DEulerFSBP
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
                        analysis_polydeg=6,
                        l2=[
                            0.11821957357197649,
                            0.15330089521538678,
                            0.4417674632047301
                        ],
                        linf=[
                            0.24280567569982958,
                            0.29130548795961936,
                            0.8847009003152442
                        ])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "TreeMesh1D Euler FSBP: elixir_euler_ec_variable_polynomial_degree.jl" setup=[
    Setup,
    TreeMesh1DEulerFSBP
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_ec_variable_polynomial_degree.jl"),
                        analysis_polydeg=6,
                        l2=[
                            0.11275325749312189,
                            0.15918993256736444,
                            0.4201116027038316
                        ],
                        linf=[
                            0.22193426272973626,
                            0.36131357156817606,
                            0.8446163262347817
                        ])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "TreeMesh1D Euler FSBP: elixir_euler_ec_er.jl" setup=[
    Setup,
    TreeMesh1DEulerFSBP
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec_er.jl"),
                        l2=[
                            0.11926705793028004,
                            0.15468940574352574,
                            0.4457918756541846
                        ],
                        linf=[
                            0.23569561716492549,
                            0.3033702248718082,
                            0.8621735653950817
                        ])
end

@testitem "TreeMesh1D Euler FSBP: relaxation conserves entropy" setup=[
    Setup,
    TreeMesh1DEulerFSBP
] tags=[:tree_part1] begin
    trixi_include(@__MODULE__, joinpath(EXAMPLES_DIR, "elixir_euler_ec_er.jl"),
                  tspan = (0.0, 0.4))

    S(u_ode) = Trixi.integrate(entropy, u_ode, semi)
    # The final step is never relaxed (gamma is forced to 1 to hit t_end exactly), so the
    # entropy is compared before it: nsteps * dt < t_end.
    nsteps, dt = 10, 0.02

    function drift(alg)
        integ = Trixi.init(semidiscretize(semi, (0.0, 0.4)), alg; dt = dt,
                           callback = nothing)
        S0 = S(integ.u)
        for _ in 1:nsteps
            Trixi.step!(integ)
        end
        return abs(S(integ.u) - S0), abs(S0)
    end

    drift_er, S0 = drift(ode_alg)                        # relaxation, from the elixir
    drift_rk, _ = drift(Trixi.CarpenterKennedy2N54())    # same machinery, no relaxation

    # Relative, and against a reference run: asserting smallness alone would pass
    # vacuously for an entropy near zero, or for a run too short to drift.
    @test drift_er / S0 < 1e-10     # round-off level  (measured 1.6e-13)
    @test drift_rk / S0 > 1e-3      # plain RK drifts  (measured 6.0e-1)
end
