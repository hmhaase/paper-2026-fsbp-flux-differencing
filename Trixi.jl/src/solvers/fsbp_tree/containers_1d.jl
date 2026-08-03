# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline eachnode(::DGFSBP, elements::TreeElementBasisContainer1D,
element) = Base.OneTo(elements.num_nodes_in_element[element][1])
@inline nnodes(::DGFSBP, elements::TreeElementBasisContainer1D,
element) = elements.num_nodes_in_element[element][1]
@inline nnodes(elements::TreeElementBasisContainer1D,
element) = elements.num_nodes_in_element[element][1]

# Create element container and initialize element data.
function init_elements(cell_ids, mesh::TreeMesh1D,
                       equations::AbstractEquations{1},
                       basis::PerCellBasis,
                       ::Type{RealT},
                       ::Type{uEltype}) where {RealT <: Real, uEltype <: Real}
    n_elements = length(cell_ids)
    CellBasisT = cell_basis_type(typeof(basis.local_basis_constructor))

    bases = Vector{CellBasisT}(undef, n_elements)
    num_nodes_in_element = Vector{NTuple{1, Int}}(undef, n_elements)
    offsets = Vector{Int}(undef, n_elements + 1)
    offsets[1] = 0
    n_vars = nvariables(equations)

    for element in 1:n_elements
        cell_id = cell_ids[element]
        dx = length_at_cell(mesh.tree, cell_id)
        cell_basis = build_cell_basis(basis.local_basis_constructor,
                                      mesh.tree.coordinates[1, cell_id], dx)

        bases[element] = cell_basis
        num_nodes_in_element[element] = (nnodes(cell_basis),)
        offsets[element + 1] = offsets[element] +
                               n_vars * ndofs_per_element(cell_basis)
    end

    return TreeElementBasisContainer{1, RealT, uEltype, CellBasisT}(bases, n_vars,
                                                                    num_nodes_in_element,
                                                                    offsets, cell_ids,
                                                                    1)
end

function init_interfaces(cell_ids, mesh::TreeMesh1D,
                         elements::TreeElementBasisContainer1D)
    n_interfaces = count_required_interfaces(mesh, cell_ids)
    interfaces = TreeInterfaceContainer1D{eltype(elements)}(n_interfaces,
                                                            nvariables(elements))
    init_interfaces!(interfaces, elements, mesh)
    return interfaces
end

function init_boundaries(cell_ids, mesh::TreeMesh1D,
                         elements::TreeElementBasisContainer1D, basis)
    n_boundaries = count_required_boundaries(mesh, cell_ids)
    boundaries = TreeBoundaryContainer1D{real(elements), eltype(elements)}(n_boundaries,
                                                                           nvariables(elements))
    init_boundaries!(boundaries, elements, mesh, basis)
    return boundaries
end

@inline function get_node_coords(elements::TreeElementBasisContainer1D, i, element)
    SVector(get_nodes(elements.bases[element])[i])
end

@inline function get_node_coords(elements::TreeElementBasisContainer1D, equations,
                                 solver::DG, indices...)
    get_node_coords(elements, indices...)
end

@inline function boundary_node_coordinate(elements::TreeElementBasisContainer1D,
                                          element,
                                          direction)
    basis = elements.bases[element]
    if direction == 1
        return SVector(boundary_coordinate(basis, Negative()))
    else
        return SVector(boundary_coordinate(basis, Positive()))
    end
end

function calc_boundary_node_coordinates!(boundaries, element, count, direction,
                                         elements::TreeElementBasisContainer1D,
                                         mesh::TreeMesh1D, basis)
    boundaries.node_coordinates[:, count] .= boundary_node_coordinate(elements, element,
                                                                      direction)
    return nothing
end
end # @muladd
