# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function max_dt(u, t, mesh::TreeMesh{1},
                constant_speed::False, equations,
                dg::DGFSBP, cache)
    # Avoid division by zero if the speed vanishes everywhere
    max_scaled_speed = nextfloat(zero(t))

    @batch reduction=(max, max_scaled_speed) for element in eachelement(dg, cache)
        basis = cache.elements.bases[element]
        u_element = u[element]
        max_lambda1 = zero(max_scaled_speed)
        for i in eachnode(basis)
            u_node = get_node_vars(u_element, equations, dg, i)
            lambda1, = max_abs_speeds(u_node, equations)
            max_lambda1 = Base.max(max_lambda1, lambda1)
        end
        Δx = length_at_cell(mesh.tree, cache.elements.cell_ids[element])
        # Use `Base.max` to prevent silent failures, as `max` from `@fastmath` doesn't propagate
        # `NaN`s properly. See https://github.com/trixi-framework/Trixi.jl/pull/2445#discussion_r2336812323
        max_scaled_speed = Base.max(max_scaled_speed,
                                    nnodes(basis) / Δx *
                                    max_lambda1)
    end

    return 1 / max_scaled_speed
end

function max_dt(u, t, mesh::TreeMesh{1},
                constant_speed::True, equations,
                dg::DGFSBP, cache)
    # Avoid division by zero if the speed vanishes everywhere,
    # e.g. for steady-state linear advection
    max_scaled_speed = nextfloat(zero(t))

    max_lambda1, = max_abs_speeds(equations)

    @batch reduction=(max, max_scaled_speed) for element in eachelement(dg, cache)
        Δx = length_at_cell(mesh.tree, cache.elements.cell_ids[element])
        # Use `Base.max` to prevent silent failures, as `max` from `@fastmath` doesn't propagate
        # `NaN`s properly. See https://github.com/trixi-framework/Trixi.jl/pull/2445#discussion_r2336812323
        max_scaled_speed = Base.max(max_scaled_speed,
                                    nnodes(cache.elements.bases[element]) / Δx *
                                    max_lambda1)
    end

    return 1 / max_scaled_speed
end
end # @muladd
