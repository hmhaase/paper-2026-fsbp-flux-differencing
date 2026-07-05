# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct TreeElementBasisContainer1D{RealT <: Real,
                                   uEltype <: Real,
                                   CellBasisT} <:
       AbstractTreeElementContainer
    bases::Vector{CellBasisT}      # [elements]
    nvars::Int
    num_nodes_in_element::Vector{NTuple{1, Int}} # 1D, so always a 1-tuple; matches WrappedArray's per-dimension representation
    offsets::Vector{Int} # 0, nvars*nnodes(bases[1]), nvars*nnodes(bases[1]) + nvars*nnodes(bases[2]), ..., nvars*nnodes(bases[1]) + ... + nvars*nnodes(bases[n])
    surface_flux_values::Array{uEltype, 3} # [variables, direction, elements]
    cell_ids::Vector{Int}                  # [elements]
    # internal `resize!`able storage
    _surface_flux_values::Vector{uEltype}
end

function TreeElementBasisContainer1D{RealT, uEltype, CellBasisT}(capacity::Integer,
                                                                 n_variables) where {
                                                                                     RealT <:
                                                                                     Real,
                                                                                     uEltype <:
                                                                                     Real,
                                                                                     CellBasisT
                                                                                     }
    nan_uEltype = convert(uEltype, NaN)

    bases = Vector{CellBasisT}(undef, capacity)
    num_nodes_in_element = Vector{NTuple{1, Int}}(undef, capacity)
    offsets = Vector{Int}(undef, capacity + 1)

    _surface_flux_values = fill(nan_uEltype, n_variables * 2 * 1 * capacity)
    surface_flux_values = unsafe_wrap(Array, pointer(_surface_flux_values),
                                      (n_variables, 2 * 1, capacity))

    cell_ids = fill(typemin(Int), capacity)

    return TreeElementBasisContainer1D{RealT, uEltype, CellBasisT}(bases,
                                                                   n_variables,
                                                                   num_nodes_in_element,
                                                                   offsets,
                                                                   surface_flux_values,
                                                                   cell_ids,
                                                                   _surface_flux_values)
end

@inline function get_node_vars(u::WrappedArray{1}, equations, solver::DGFSBP,
                               i::Int, element::Int)
    idx_base = u.nvars * (i - 1) + u.offsets[element]
    SVector(ntuple(@inline(v->@inbounds(u.a[v + idx_base])),
                   Val(nvariables(equations))))
end

@inline Base.real(::TreeElementBasisContainer1D{RealT, uEltype, CellBasisT}) where {
RealT <: Real,
uEltype <: Real,
CellBasisT
} = RealT

@inline function nelements(elements::TreeElementBasisContainer1D)
    length(elements.bases)
end

@inline eachnode(::DGFSBP, elements::TreeElementBasisContainer1D, element) = Base.OneTo(elements.num_nodes_in_element[element][1])
@inline nnodes(::DGFSBP, elements::TreeElementBasisContainer1D, element) = elements.num_nodes_in_element[element][1]
@inline nnodes(elements::TreeElementBasisContainer1D, element) = elements.num_nodes_in_element[element][1]

"""
    eachelement(elements::TreeElementBasisContainer1D)

Return an iterator over the indices that specify the location in relevant data structures
for the elements in `elements`.
In particular, not the elements themselves are returned.
"""
@inline function eachelement(elements::TreeElementBasisContainer1D)
    Base.OneTo(nelements(elements))
end

num_total_nodes(elements::TreeElementBasisContainer1D) = elements.offsets[end] ÷
                                                         elements.nvars

quadrature_weights(::DGFSBP, elements::TreeElementBasisContainer1D, element) = quadrature_weights(elements.bases[element])

# Create element container and initialize element data
function init_elements(cell_ids, mesh::TreeMesh1D,
                       equations::AbstractEquations{1},
                       basis::PerCellBasis{CellBasisConstructorT},
                       ::Type{RealT},
                       ::Type{uEltype}) where
         {
          RealT <: Real,
          uEltype <: Real,
          CellBasisT <: CellBasis{RealT},
          CellBasisConstructorT <: CellBasisConstructor{CellBasisT}
          }
    # Initialize container
    n_elements = length(cell_ids)
    elements = TreeElementBasisContainer1D{RealT, uEltype, CellBasisT}(n_elements,
                                                                       nvariables(equations))

    init_elements!(elements, cell_ids, mesh, basis)
    return elements
end

function init_elements!(elements, cell_ids, mesh::TreeMesh1D, basis::PerCellBasis)
    # Store cell ids
    elements.cell_ids .= cell_ids
    elements.offsets[begin] = 0

    n_vars = nvariables(elements)

    for element in eachelement(elements)
        # Get cell id
        cell_id = cell_ids[element]

        # Get cell length
        dx = length_at_cell(mesh.tree, cell_id)

        basis_ref = Ref(elements.bases, element)
        init_local_basis!(basis.local_basis_constructor, basis_ref,
                          mesh.tree.coordinates[1, cell_id], dx)
        elements.offsets[element + 1] = elements.offsets[element] +
                                        n_vars * nnodes(basis_ref[])
        elements.num_nodes_in_element[element] = (nnodes(basis_ref[]),)
    end

    return elements
end

function init_interfaces(cell_ids, mesh::TreeMesh1D,
                         elements::TreeElementBasisContainer1D)
    # Initialize container
    n_interfaces = count_required_interfaces(mesh, cell_ids)
    interfaces = TreeInterfaceContainer1D{eltype(elements)}(n_interfaces,
                                                            nvariables(elements))

    # Connect elements with interfaces
    init_interfaces!(interfaces, elements, mesh)
    return interfaces
end

# Create boundaries container and initialize boundary data in `elements`.
function init_boundaries(cell_ids, mesh::TreeMesh1D,
                         elements::TreeElementBasisContainer1D, basis)
    # Initialize container
    n_boundaries = count_required_boundaries(mesh, cell_ids)
    boundaries = TreeBoundaryContainer1D{real(elements), eltype(elements)}(n_boundaries,
                                                                           nvariables(elements))

    # Connect elements with boundaries
    init_boundaries!(boundaries, elements, mesh, basis)
    return boundaries
end

@inline function boundary_node_coordinate(elements::TreeElementBasisContainer1D, element,
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

function wrap_array(u_ode::AbstractVector, mesh::AbstractMesh, equations,
                    dg::DGFSBP, cache)
    WrappedArray(Val(ndims(equations)), nvariables(equations),
                 cache.elements.num_nodes_in_element,
                 cache.elements.offsets, u_ode)
end

function allocate_coefficients(::AbstractMesh, equations, ::DGFSBP, cache)
    # We must allocate a `Vector` in order to be able to `resize!` it (AMR).
    # cf. wrap_array
    u_ode = fill(zero(eltype(cache.elements)),
                nvariables(equations) * num_total_nodes(cache.elements))
    return u_ode
end

@inline function get_node_coords(elements::TreeElementBasisContainer1D, indices...)
    x1 = get_nodes(elements.bases[indices[end]])[indices[begin:(end - 1)]...]
    SVector(x1)
end

@inline function get_node_coords(elements::TreeElementBasisContainer1D, equations,
                                 solver::DG, indices...)
    get_node_coords(elements, indices...)
end

@inline nelements(mesh, dg::DGFSBP, cache) = nelements(cache.elements)
@inline function ndofsglobal(mesh, dg::DGFSBP, cache)
    num_total_nodes(cache.elements)
end

@inline function ndofs(mesh::MeshesDGSEM, dg::DGFSBP, cache)
    num_total_nodes(cache.elements)
end
end # @muladd
