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

@testitem "TreeMesh2D Advection FSBP: elixir_advection_basic.jl" setup=[
    Setup,
    TreeMesh2DFSBP
] tags=[:tree_part1] begin
    # `analysis_polydeg = nothing` in 2D, so `l2` differs from the DGSEM value.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_fsbp",
                                 "elixir_advection_basic.jl"),
                        l2=[1.2574886515501355e-5],
                        linf=[6.627000273806694e-5])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
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

@testitem "TreeMesh2D Euler FSBP: quasi-1D reduces to the 1D solver" setup=[Setup] tags=[:tree_part1] begin
    using OrdinaryDiffEqLowStorageRK

    # Data invariant along one axis, with no velocity component across it, decouples the
    # 2D scheme into independent 1D problems, so the 1D solver is the reference.
    #
    # Smooth, since the entropy-conservative surface flux carries no dissipation, but
    # nonlinear: density, velocity and pressure all vary.
    prim(s) = (1 + 0.3 * sinpi(s), 0.2 * cospi(s), 1 + 0.4 * sinpi(s))
    ic_1d(x, t, eq::CompressibleEulerEquations1D) = prim2cons(SVector(prim(x[1])...), eq)
    function ic_2d(axis)
        return function (x, t, eq::CompressibleEulerEquations2D)
            rho, v, p = prim(x[axis])
            return prim2cons(SVector(rho, axis == 1 ? v : 0.0, axis == 1 ? 0.0 : v, p), eq)
        end
    end

    make_mesh(coords...) = TreeMesh(coords...; initial_refinement_level = 3,
                                    n_cells_max = 10^4, periodicity = true)
    centres(semi, axis) = [semi.mesh.tree.coordinates[axis, cell_id]
                           for cell_id in semi.cache.elements.cell_ids]

    # Fixed `dt`: for Euler the transverse wave speed is the sound speed, not zero, so a
    # `StepsizeCallback` would pick different steps in 2D than in 1D.
    function evolve(mesh, equations, ic)
        solver = DGFSBP(LobattoLegendreCellBasisConstructor{Float64}((args...) -> 3);
                        surface_flux = flux_ranocha,
                        volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))
        semi = SemidiscretizationHyperbolic(mesh, equations, ic, solver;
                                            boundary_conditions = boundary_condition_periodic)
        sol = solve(semidiscretize(semi, (0.0, 0.4)),
                    CarpenterKennedy2N54(williamson_condition = false);
                    dt = 0.01, adaptive = false, save_everystep = false,
                    callback = nothing)
        # `analysis_polydeg` interpolation is not implemented for 2D DGFSBP
        l2, _ = AnalysisCallback(semi, interval = typemax(Int),
                                 analysis_polydeg = nothing)(sol)
        return semi, Trixi.wrap_array(sol.u[end], semi), l2
    end

    semi_1d, u_1d, l2_1d = evolve(make_mesh(-1.0, 1.0), CompressibleEulerEquations1D(1.4),
                                  ic_1d)
    centres_1d = centres(semi_1d, 1)

    for axis in (1, 2)
        semi_2d, u_2d, l2_2d = evolve(make_mesh((-1.0, -1.0), (1.0, 1.0)),
                                      CompressibleEulerEquations2D(1.4), ic_2d(axis))
        # 1D (rho, rho*v, rho*e) -> 2D variables; the transverse momentum stays zero
        vars = axis == 1 ? (1, 2, 4) : (1, 3, 4)

        # Transverse-invariant data gives the same volume-normalised L2 norms in both runs
        @test l2_2d[collect(vars)]≈l2_1d atol=1e-14

        # Every 2D element is compared against the 1D element with the same centre, on
        # every transverse line, so this also pins transverse invariance.
        partner = [findfirst(≈(c), centres_1d) for c in centres(semi_2d, axis)]
        deviation = maximum(Trixi.eachelement(semi_2d.solver, semi_2d.cache)) do element
            uel = u_2d[element]
            maximum(abs(uel[v, i, j] - u_1d[k, axis == 1 ? i : j, partner[element]])
                    for (k, v) in enumerate(vars), j in axes(uel, 3), i in axes(uel, 2))
        end
        @test deviation < 1e-13

        momentum = maximum(Trixi.eachelement(semi_2d.solver, semi_2d.cache)) do element
            maximum(abs, view(u_2d[element], axis == 1 ? 3 : 2, :, :))
        end
        @test momentum < 1e-13
    end
end
