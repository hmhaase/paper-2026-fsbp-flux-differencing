@testsnippet TreeMesh1DAdvectionFSBP begin
    EXAMPLES_DIR = joinpath(examples_dir(), "tree_1d_fsbp")
end

@testitem "TreeMesh1D Advection FSBP: elixir_advection_boundary_layer.jl" setup=[
    Setup,
    TreeMesh1DAdvectionFSBP
] tags=[:tree_part1] begin
    # l2/linf look large in absolute terms, but the exact solution itself spans
    # 1.0 to exp(2*pi) ~ 535.5 on this domain; the relative error is ~0.5%.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_boundary_layer.jl"),
        l2 = [0.5180527023144795],
        linf = [2.625579703349274])

    # LinearScalarAdvectionEquation1D has a constant wave speed, so this elixir
    # exercises the constant_speed::True branch of max_dt for DGFSBP.
    @test Trixi.have_constant_speed(equations) === Trixi.True()
end
