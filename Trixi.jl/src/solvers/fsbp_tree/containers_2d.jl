# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct TreeElementBasisContainer2D{RealT <: Real,
                                   uEltype <: Real,
                                   CellBasisT} <:
       AbstractTreeElementContainer
    bases::Vector{CellBasisT}                     # [elements]
    nvars::Int
    num_nodes_in_element::Vector{NTuple{2, Int}}  # (nx, ny) per element
    offsets::Vector{Int}                          # start of each element in the flat vector
    surface_flux_values::Array{uEltype, 4}        # [variables, face node, direction (1:4), elements]
    cell_ids::Vector{Int}                         # [elements]
    _surface_flux_values::Vector{uEltype}         # internal `resize!`able storage
end

# `n_face_nodes` (max node count over all elements/axes) sizes the face dimension of `surface_flux_values`.
function TreeElementBasisContainer2D{RealT, uEltype, CellBasisT}(bases, nvars,
                                                                 num_nodes_in_element,
                                                                 offsets, cell_ids,
                                                                 n_face_nodes) where {RealT <:
                                                                                      Real,
                                                                                      uEltype <:
                                                                                      Real,
                                                                                      CellBasisT}
    capacity = length(bases)
    nan_uEltype = convert(uEltype, NaN)

    _surface_flux_values = fill(nan_uEltype, nvars * n_face_nodes * 4 * capacity)
    surface_flux_values = unsafe_wrap(Array, pointer(_surface_flux_values),
                                      (nvars, n_face_nodes, 4, capacity))

    return TreeElementBasisContainer2D{RealT, uEltype, CellBasisT}(bases, nvars,
                                                                   num_nodes_in_element,
                                                                   offsets,
                                                                   surface_flux_values,
                                                                   collect(Int, cell_ids),
                                                                   _surface_flux_values)
end

@inline Base.real(::TreeElementBasisContainer2D{RealT}) where {RealT <: Real} = RealT

@inline nelements(elements::TreeElementBasisContainer2D) = length(elements.bases)
@inline eachelement(elements::TreeElementBasisContainer2D) = Base.OneTo(nelements(elements))

num_total_nodes(elements::TreeElementBasisContainer2D) = elements.offsets[end] ÷
                                                         elements.nvars

@inline eachnode(::DGFSBP, elements::TreeElementBasisContainer2D, element) = CartesianIndices(elements.num_nodes_in_element[element])
@inline nnodes(::DGFSBP, elements::TreeElementBasisContainer2D, element) = prod(elements.num_nodes_in_element[element])

# Create element container and initialize element data.
function init_elements(cell_ids, mesh::TreeMesh2D,
                       equations::AbstractEquations{2},
                       basis::PerCellBasis,
                       ::Type{RealT},
                       ::Type{uEltype}) where {RealT <: Real, uEltype <: Real}
    n_elements = length(cell_ids)
    # One tensor-product basis per element; a bare 1D constructor is wrapped isotropically.
    tensor_ctor = as_tensor(basis.local_basis_constructor, Val(2))
    CellBasisT = cell_basis_type(typeof(tensor_ctor))

    bases = Vector{CellBasisT}(undef, n_elements)
    num_nodes_in_element = Vector{NTuple{2, Int}}(undef, n_elements)
    offsets = Vector{Int}(undef, n_elements + 1)
    offsets[1] = 0
    n_vars = nvariables(equations)
    n_face_nodes = 0

    for element in 1:n_elements
        cell_id = cell_ids[element]
        dx = length_at_cell(mesh.tree, cell_id)
        pos = view(mesh.tree.coordinates, :, cell_id)
        # TreeMesh cells are square: identical width on both axes
        tp = build_cell_basis(tensor_ctor, pos, (dx, dx))

        bases[element] = tp
        nn = map(nnodes, tp.bases)
        num_nodes_in_element[element] = nn
        offsets[element + 1] = offsets[element] + n_vars * ndofs_per_element(tp)
        n_face_nodes = max(n_face_nodes, maximum(nn))
    end

    return TreeElementBasisContainer2D{RealT, uEltype, CellBasisT}(bases, n_vars,
                                                                   num_nodes_in_element,
                                                                   offsets, cell_ids,
                                                                   n_face_nodes)
end

# Reuse the DGSEM 2D interface/boundary containers; face-node count taken from `surface_flux_values`.
function init_interfaces(cell_ids, mesh::TreeMesh2D,
                         elements::TreeElementBasisContainer2D)
    n_interfaces = count_required_interfaces(mesh, cell_ids)
    n_face_nodes = size(elements.surface_flux_values, 2)
    interfaces = TreeInterfaceContainer2D{eltype(elements)}(n_interfaces,
                                                            nvariables(elements),
                                                            n_face_nodes)
    init_interfaces!(interfaces, elements, mesh)
    return interfaces
end

function init_boundaries(cell_ids, mesh::TreeMesh2D,
                         elements::TreeElementBasisContainer2D, basis)
    n_boundaries = count_required_boundaries(mesh, cell_ids)
    n_face_nodes = size(elements.surface_flux_values, 2)
    boundaries = TreeBoundaryContainer2D{real(elements), eltype(elements)}(n_boundaries,
                                                                           nvariables(elements),
                                                                           n_face_nodes)
    init_boundaries!(boundaries, elements, mesh, basis)
    return boundaries
end

@inline function get_node_coords(elements::TreeElementBasisContainer2D, i, j, element)
    bx, by = elements.bases[element].bases
    return SVector(get_nodes(bx)[i], get_nodes(by)[j])
end

@inline function get_node_coords(elements::TreeElementBasisContainer2D, equations,
                                 solver::DG, indices...)
    get_node_coords(elements, indices...)
end

# Fill the physical coordinates of one boundary's face nodes from the element's tensor basis:
# the normal axis sits at the element boundary, the tangential axis runs over its nodes.
function calc_boundary_node_coordinates!(boundaries, element, count, direction,
                                         elements::TreeElementBasisContainer2D,
                                         mesh::TreeMesh2D, basis)
    bx, by = elements.bases[element].bases
    bnd = boundaries.node_coordinates
    if direction == 1 || direction == 2       # x-face: tangential runs over y
        x = boundary_coordinate(bx, direction == 1 ? Negative() : Positive())
        for j in eachnode(by)
            bnd[1, j, count] = x
            bnd[2, j, count] = get_nodes(by)[j]
        end
    else                                       # y-face: tangential runs over x
        y = boundary_coordinate(by, direction == 3 ? Negative() : Positive())
        for i in eachnode(bx)
            bnd[1, i, count] = get_nodes(bx)[i]
            bnd[2, i, count] = y
        end
    end
    return nothing
end
end # @muladd
