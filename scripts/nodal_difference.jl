# Difference of two solutions of the same 2D TreeMesh problem, taken at the target's nodes
# rather than on a plot grid, so no FSBP reconstruction enters. The mesh must be uniform.

using Trixi

solver_of(semi) = Trixi.mesh_equations_solver_cache(semi)[3]
cache_of(semi) = Trixi.mesh_equations_solver_cache(semi)[4]
isfsbp(semi) = solver_of(semi).basis isa Trixi.PerCellBasis

# Whether the basis spans polynomials of degree `nnodes - 1`, so that Lagrange interpolation
# from its nodes is exact. Checked on the operator itself, since its type does not tell.
function is_polynomial(semi)
    isfsbp(semi) || return true
    basis = first(cache_of(semi).elements.bases).bases[1]
    nodes = Trixi.get_nodes(basis)
    probe = collect(range(Trixi.boundary_coordinate(basis, Trixi.Negative()),
                          Trixi.boundary_coordinate(basis, Trixi.Positive()),
                          length = 2 * length(nodes) + 1))
    # The basis interpolates through a closure the elixir built, newer than this method.
    return isapprox(Base.invokelatest(Trixi.interpolation_matrix, basis, probe),
                    Trixi.polynomial_interpolation_matrix(nodes, probe); atol = 1e-8)
end

node_range(idx, n) = ((idx - 1) * n + 1):(idx * n)

# One element's nodal values as `[variable, i, j]`, for either solver family.
function element_slicer(semi)
    isfsbp(semi) && return (u, element) -> u[element]
    return (u, element) -> view(u, :, :, :, element)
end

# Node positions of a run's basis on the reference cell [-1, 1].
function reference_nodes(semi)
    isfsbp(semi) || return solver_of(semi).basis.nodes
    bx = first(cache_of(semi).elements.bases).bases[1]
    xl = Trixi.boundary_coordinate(bx, Trixi.Negative())
    xr = Trixi.boundary_coordinate(bx, Trixi.Positive())
    nodes = Trixi.get_nodes(bx)          # bound first: `@.` would broadcast the call
    return @. 2 * (nodes - xl) / (xr - xl) - 1
end

# Each element's column and row in the uniform grid, and the node coordinates per axis.
function element_layout(semi)
    tree = semi.mesh.tree
    elements = cache_of(semi).elements
    n = length(reference_nodes(semi))
    h = Trixi.length_at_cell(tree, first(elements.cell_ids))
    ncells = round(Int, tree.length_level_0 / h)
    origin = tree.center_level_0 .- tree.length_level_0 / 2
    columns, rows = Int[], Int[]
    xs, ys = zeros(ncells * n), zeros(ncells * n)
    for element in Trixi.eachelement(elements)
        cell_id = elements.cell_ids[element]
        Trixi.length_at_cell(tree, cell_id) == h ||
            error("element $element has a different width; the mesh is not uniform")
        column, row = round.(Int,
                             (Trixi.cell_coordinates(tree, cell_id) .- origin) ./ h .+ 0.5)
        push!(columns, column)
        push!(rows, row)
        xs[node_range(column, n)], ys[node_range(row, n)] = node_coordinates(semi, element, n)
    end
    return (; columns, rows, xs, ys, n, ncells)
end

# FSBP bases hold their physical nodes; DGSEM stores them as `[dim, i, j, element]`.
function node_coordinates(semi, element, n)
    elements = cache_of(semi).elements
    if isfsbp(semi)
        bx, by = elements.bases[element].bases
        Trixi.nnodes(bx) == n && Trixi.nnodes(by) == n ||
            error("element $element has $(Trixi.nnodes(bx)) x $(Trixi.nnodes(by)) nodes, not $n x $n")
        return Trixi.get_nodes(bx), Trixi.get_nodes(by)
    end
    x = elements.node_coordinates
    return x[1, :, 1, element], x[2, 1, :, element]
end

"""
    nodal_difference(source, target; variable = 1)

`source` and `target` are named tuples `(; semi, u_ode)`. Returns `(; x, y, values)` with
`values[i, j]` the source, interpolated to the target's node `(x[i], y[j])`, minus the
target, in conservative variable `variable`. `source` must have a polynomial basis.
"""
function nodal_difference(source, target; variable = 1)
    is_polynomial(source.semi) ||
        error("source must be the polynomial operator, else interpolating it loses information")
    cache_of(source.semi).elements.cell_ids == cache_of(target.semi).elements.cell_ids ||
        error("element orderings differ")
    (; columns, rows, xs, ys, n, ncells) = element_layout(target.semi)

    V = Trixi.polynomial_interpolation_matrix(reference_nodes(source.semi),
                                              reference_nodes(target.semi))
    source_slice, target_slice = element_slicer(source.semi), element_slicer(target.semi)
    u_source = Trixi.wrap_array(source.u_ode, source.semi)
    u_target = Trixi.wrap_array(target.u_ode, target.semi)

    out = zeros(ncells * n, ncells * n)
    for element in eachindex(columns)
        src = source_slice(u_source, element)
        # `multiply_dimensionwise` runs under `@turbo` without bounds checks, so a
        # node count that does not match `V` would give wrong values, not an error.
        size(src, 2) == size(V, 2) && size(src, 3) == size(V, 2) ||
            error("source element $element has a node count V was not built for")
        difference = Trixi.multiply_dimensionwise(V, src)[variable, :, :] .-
                     target_slice(u_target, element)[variable, :, :]
        out[node_range(columns[element], n), node_range(rows[element], n)] = difference
    end
    return (; x = xs, y = ys, values = out)
end

"""
    nodal_values(run; variable = 1)

`run` is a named tuple `(; semi, u_ode)`. Returns `(; x, y, values)` with `values[i, j]`
conservative variable `variable` at the run's node `(x[i], y[j])`.
"""
function nodal_values(run; variable = 1)
    (; columns, rows, xs, ys, n, ncells) = element_layout(run.semi)
    slice = element_slicer(run.semi)
    u = Trixi.wrap_array(run.u_ode, run.semi)

    out = zeros(ncells * n, ncells * n)
    for element in eachindex(columns)
        out[node_range(columns[element], n), node_range(rows[element], n)] =
            slice(u, element)[variable, :, :]
    end
    return (; x = xs, y = ys, values = out)
end

"""
    nodal_error(run, t; variable = 1)

As `nodal_values`, minus the exact value at each node, the exact solution being the
semidiscretization's initial condition at `t`.
"""
function nodal_error(run, t; variable = 1)
    (; x, y, values) = nodal_values(run; variable)
    (; initial_condition, equations) = run.semi
    exact = [initial_condition(SVector(xi, yj), t, equations)[variable] for xi in x, yj in y]
    return (; x, y, values = values .- exact)
end
