@testsnippet TreeMesh2DFSBP begin
    EXAMPLES_DIR = examples_dir()
end

@testitem "TreeMesh2D Advection FSBP: FSBP vs DGSEM comparison" setup=[
    Setup,
    TreeMesh2DFSBP
] tags=[:tree_part1] begin
    using LinearAlgebra: norm

    # A Lobatto-Legendre `DGFSBP` tensor-product basis is the same operator as DGSEM, so on
    # the same (uniform, isotropic) mesh the two must agree up to floating point round-off.
    trixi_include(@__MODULE__,
        joinpath(EXAMPLES_DIR, "tree_2d_dgsem", "elixir_advection_basic.jl"))
    u_dgsem = copy(sol.u[end])

    trixi_include(@__MODULE__,
        joinpath(EXAMPLES_DIR, "tree_2d_fsbp", "elixir_advection_basic.jl"))
    u_fsbp = copy(sol.u[end])

    @test norm(u_fsbp - u_dgsem, Inf) < 1e-12
end

@testitem "TreeMesh2D Euler EC FSBP: FSBP vs DGSEM comparison" setup=[
    Setup,
    TreeMesh2DFSBP
] tags=[:tree_part1] begin
    using LinearAlgebra: norm

    # Same check for the flux-differencing (entropy-conservative) volume integral.
    trixi_include(@__MODULE__,
        joinpath(EXAMPLES_DIR, "tree_2d_dgsem", "elixir_euler_ec.jl"),
        initial_refinement_level = 3, tspan = (0.0, 0.2))
    u_dgsem = copy(sol.u[end])

    trixi_include(@__MODULE__,
        joinpath(EXAMPLES_DIR, "tree_2d_fsbp", "elixir_euler_ec.jl"),
        initial_refinement_level = 3, tspan = (0.0, 0.2))
    u_fsbp = copy(sol.u[end])

    @test norm(u_fsbp - u_dgsem, Inf) < 1e-12
end

@testitem "TreeMesh2D Euler source terms nonperiodic FSBP: FSBP vs DGSEM comparison" setup=[
    Setup,
    TreeMesh2DFSBP
] tags=[:tree_part1] begin
    using LinearAlgebra: norm

    # Non-periodic Dirichlet boundaries and source terms: Lobatto DGFSBP must still match
    # DGSEM to round-off.
    trixi_include(@__MODULE__,
        joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
            "elixir_euler_source_terms_nonperiodic.jl"),
        initial_refinement_level = 2, tspan = (0.0, 0.2))
    u_dgsem = copy(sol.u[end])

    trixi_include(@__MODULE__,
        joinpath(EXAMPLES_DIR, "tree_2d_fsbp",
            "elixir_euler_source_terms_nonperiodic.jl"),
        initial_refinement_level = 2, tspan = (0.0, 0.2))
    u_fsbp = copy(sol.u[end])

    @test norm(u_fsbp - u_dgsem, Inf) < 1e-12
end
