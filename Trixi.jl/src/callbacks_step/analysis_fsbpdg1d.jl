function create_cache_analysis(analyzer, mesh::TreeMesh{1},
                               equations, dg::DGFSBP, cache,
                               RealT, uEltype)
    if isnothing(analyzer.analysis_polydeg)
        (;)
    else
        nnodes_ = analyzer.analysis_polydeg + 1

        RealT = real(dg)
        nodes, weights = gauss_lobatto_nodes_weights(nnodes_, RealT)

        # pre-allocate buffers
        # We use `StrideArray`s here since these buffers are used in performance-critical
        # places and the additional information passed to the compiler makes them faster
        # than native `Array`s.
        u_local = StrideArray(undef, uEltype,
                              StaticInt(nvariables(equations)),
                              StaticInt(analyzer.analysis_polydeg + 1))
        x_local = StrideArray(undef, RealT,
                              StaticInt(ndims(equations)),
                              StaticInt(analyzer.analysis_polydeg + 1))

        (; u_local, x_local, nodes, weights)
    end
end

function integrate_via_indices(func::Func, u,
                               mesh::TreeMesh{1}, equations, dg::DGFSBP, cache,
                               args...; normalize = true) where {Func}

    # Initialize integral with zeros of the right shape
    integral = zero(func(u, 1, 1, equations, dg, args...))

    # Use quadrature to numerically integrate over entire domain
    @batch reduction=(+, integral) for element in eachelement(dg, cache)
        basis = cache.elements.bases[element]
        weights = quadrature_weights(basis)
        for i in eachnode(basis)
            integral += weights[i] *
                        func(u, i, element, equations, dg, args...)
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
                          mesh::TreeMesh{1}, equations, exact_solution,
                          dg::DGFSBP, cache, cache_analysis)

Calculate L2 and LInf error for func(u_exact) and func(u), where
`u_exact(x, t) = exact_solution(x, t, equations)`.
The L2 and LInf norms are approximated using some quadrature rule given by the analyzer.
"""
function calc_error_norms(func, u, t, analyzer,
                          mesh::TreeMesh{1}, equations, exact_solution,
                          dg::DGFSBP, cache, cache_analysis)

    # Set up data structures
    l2_error = zero(func(get_node_vars(u[1], equations, dg, 1), equations))
    linf_error = copy(l2_error)

    if isnothing(analyzer.analysis_polydeg)
        # Iterate over all elements for error calculations
        for element in eachelement(dg, cache)
            basis = cache.elements.bases[element]
            u_element = u[element]
            weights = quadrature_weights(basis)

            # for i in eachnode(analyzer)
            for i in eachnode(basis)
                u_exact = exact_solution(get_node_coords(cache.elements, equations, dg,
                                                         i,
                                                         element), t,
                                         equations)
                diff = func(u_exact, equations) -
                       func(get_node_vars(u_element, equations, dg, i), equations)
                l2_error += diff .^ 2 * weights[i]
                linf_error = @. max(linf_error, abs(diff))
            end
        end
    else
        @unpack u_local, x_local, nodes, weights = cache_analysis

        # Iterate over all elements for error calculations
        for element in eachelement(dg, cache)
            cell_id = cache.elements.cell_ids[element]
            dx2 = length_at_cell(mesh.tree, cell_id) / 2
            p = cell_coordinates(mesh.tree, cell_id)[1] # cell center
            x_local[1, :] .= nodes .* dx2 .+ p

            # TODO: Make this more efficient
            vandermonde = interpolation_matrix(cache.elements.bases[element], x_local)
            multiply_dimensionwise!(u_local, vandermonde, u[element])

            for i in 1:length(nodes)
                u_exact = exact_solution(get_node_coords(x_local, equations, dg, i), t,
                                         equations)
                diff = func(u_exact, equations) -
                       func(get_node_vars(u_local, equations, dg, i), equations)
                l2_error += diff .^ 2 * (weights[i] * dx2)
                linf_error = @. max(linf_error, abs(diff))
            end
        end
    end

    # For L2 error, divide by total volume
    total_volume_ = total_volume(mesh)
    l2_error = @. sqrt(l2_error / total_volume_)

    return l2_error, linf_error
end
