@testsnippet TreeMesh1DBurgersFSBP begin
    EXAMPLES_DIR = examples_dir()
end

@testitem "TreeMesh1D Burgers FSBP: FSBP vs DGSEM comparison" setup=[
    Setup,
    TreeMesh1DBurgersFSBP
] tags=[:tree_part1] begin
    using LinearAlgebra: norm

    trixi_include(@__MODULE__,
        joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
            "elixir_burgers_linear_stability.jl"),
        tspan=(0.0, 0.5))

    # get solution vector at 0.5
    u_dgsem = copy(sol.u[end])

    trixi_include(@__MODULE__,
        joinpath(EXAMPLES_DIR, "tree_1d_fsbp",
            "elixir_burgers_linear_stability.jl"),
        tspan=(0.0, 0.5))

    # get solution vector at 0.5
    u_fsbp = copy(sol.u[end])

    # apart from floating point inaccuracies the results should be equal
    @test norm(u_fsbp - u_dgsem, Inf) < 1e-12
end

@testitem "TreeMesh1D Burgers FSBP: DGFSBP weak form vs DGSEM" setup=[
    Setup,
    TreeMesh1DBurgersFSBP
] tags=[:tree_part1] begin
    using LinearAlgebra: norm

    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "tree_1d_fsbp", "elixir_burgers_basic.jl"),
                  tspan = (0.0, 0.5), analysis_polydeg = 6,
                  solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs))
    u_dgsem = copy(sol.u[end])

    polydeg = 3
    solver_lobatto_like = DGFSBP(LobattoLegendreCellBasisConstructor{Float64}((args...) -> polydeg);
                                  surface_flux = flux_lax_friedrichs,
                                  volume_integral = VolumeIntegralWeakForm())
    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "tree_1d_fsbp", "elixir_burgers_basic.jl"),
                  tspan = (0.0, 0.5), analysis_polydeg = 6, solver = solver_lobatto_like)
    u_lobatto_like = copy(sol.u[end])

    @test norm(u_lobatto_like - u_dgsem, Inf) < 1e-12
end

@testitem "TreeMesh1D Burgers FSBP: elixir_burgers_basic.jl: error norms match DGSEM (analysis_polydeg=6)" setup=[
    Setup,
    TreeMesh1DBurgersFSBP
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_fsbp", "elixir_burgers_basic.jl"),
        analysis_polydeg = 6,
        l2 = [2.967470209082194e-5],
        linf = [0.00016152468882624227])

    # analysis_polydeg=6 matches DGSEM's own default (2 * polydeg); at that degree
    # the FSBP and DGSEM error norms should agree, not just this fixed reference.
    l2_fsbp, linf_fsbp = analysis_callback(sol)

    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "tree_1d_dgsem", "elixir_burgers_basic.jl"))
    l2_dgsem, linf_dgsem = analysis_callback(sol)

    @test l2_fsbp ≈ l2_dgsem
    @test linf_fsbp ≈ linf_dgsem
end
