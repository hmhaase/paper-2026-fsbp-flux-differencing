# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function create_cache_analysis(analyzer, mesh::TreeMesh{2},
                               equations, dg::DGFSBP, cache,
                               RealT, uEltype)
    if isnothing(analyzer.analysis_polydeg)
        (;)
    else
        error("analysis_polydeg interpolation is not yet implemented for 2D DGFSBP")
    end
end

function integrate_via_indices(func::Func, u,
                               mesh::TreeMesh{2}, equations, dg::DGFSBP, cache,
                               args...; normalize = true) where {Func}
    # Initialize integral with zeros of the right shape
    integral = zero(func(u, 1, 1, 1, equations, dg, args...))

    # Use tensor quadrature (w_i * w_j) to numerically integrate over the domain
    @batch reduction=(+, integral) for element in eachelement(dg, cache)
        bx, by = cache.elements.bases[element].bases
        wx = quadrature_weights(bx)
        wy = quadrature_weights(by)
        for j in eachnode(by), i in eachnode(bx)
            integral += wx[i] * wy[j] *
                        func(u, i, j, element, equations, dg, args...)
        end
    end

    # Normalize with total volume
    if normalize
        integral = integral / total_volume(mesh)
    end

    return integral
end

"""
    calc_error_norms(func, u, t, analyzer,
                          mesh::TreeMesh{2}, equations, exact_solution,
                          dg::DGFSBP, cache, cache_analysis)

Calculate L2 and LInf error for func(u_exact) and func(u), where
`u_exact(x, t) = exact_solution(x, t, equations)`.
The L2 and LInf norms are approximated using some quadrature rule given by the analyzer.
"""
function calc_error_norms(func, u, t, analyzer,
                          mesh::TreeMesh{2}, equations, exact_solution,
                          dg::DGFSBP, cache, cache_analysis)
    l2_error = zero(func(get_node_vars(u[1], equations, dg, 1, 1), equations))
    linf_error = copy(l2_error)

    if !isnothing(analyzer.analysis_polydeg)
        error("analysis_polydeg interpolation is not yet implemented for 2D DGFSBP")
    end

    for element in eachelement(dg, cache)
        bx, by = cache.elements.bases[element].bases
        wx = quadrature_weights(bx)
        wy = quadrature_weights(by)
        u_element = u[element]

        for j in eachnode(by), i in eachnode(bx)
            x_node = get_node_coords(cache.elements, i, j, element)
            u_exact = exact_solution(x_node, t, equations)
            diff = func(u_exact, equations) -
                   func(get_node_vars(u_element, equations, dg, i, j), equations)
            l2_error += diff .^ 2 * (wx[i] * wy[j])
            linf_error = @. max(linf_error, abs(diff))
        end
    end

    # For L2 error, divide by total volume
    total_volume_ = total_volume(mesh)
    l2_error = @. sqrt(l2_error / total_volume_)

    return l2_error, linf_error
end
end # @muladd
