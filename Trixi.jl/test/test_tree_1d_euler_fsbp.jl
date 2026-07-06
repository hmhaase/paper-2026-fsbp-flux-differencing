@testsnippet TreeMesh1DEulerFSBP begin
    EXAMPLES_DIR = joinpath(examples_dir(), "tree_1d_fsbp")
end

@testitem "TreeMesh1D Euler FSBP: elixir_euler_ec.jl" setup=[
    Setup,
    TreeMesh1DEulerFSBP
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
        analysis_polydeg = 6,
        l2 = [
            0.11821957357197649,
            0.15330089521538678,
            0.4417674632047301
        ],
        linf = [
            0.24280567569982958,
            0.29130548795961936,
            0.8847009003152442
        ])

    # TODO: Fix allocations
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
