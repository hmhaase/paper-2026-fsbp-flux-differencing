# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# `NDIMSP2` is `NDIMS + 2`, the rank of `surface_flux_values`; Julia cannot compute it
# inside the type, so it is carried as a parameter (as in `P4estElementContainer`).
struct TreeElementBasisContainer{NDIMS, RealT <: Real, uEltype <: Real, CellBasisT,
                                 NDIMSP2} <: AbstractTreeElementContainer
    bases::Vector{CellBasisT}                        # [elements]
    nvars::Int
    num_nodes_in_element::Vector{NTuple{NDIMS, Int}} # per axis, per element
    offsets::Vector{Int}                             # element starts in the flat vector
    # [variables, face node..., direction, elements]
    surface_flux_values::Array{uEltype, NDIMSP2}
    cell_ids::Vector{Int}                            # [elements]
    _surface_flux_values::Vector{uEltype}             # internal `resize!`able storage
end

const TreeElementBasisContainer1D = TreeElementBasisContainer{1}
const TreeElementBasisContainer2D = TreeElementBasisContainer{2}

# `n_face_nodes` (max node count over all elements/axes) sizes the face dimensions of
# `surface_flux_values`; in 1D there are none and it is ignored.
function TreeElementBasisContainer{NDIMS, RealT, uEltype, CellBasisT}(bases, nvars,
                                                                      num_nodes_in_element,
                                                                      offsets, cell_ids,
                                                                      n_face_nodes) where {
                                                                                           NDIMS,
                                                                                           RealT <:
                                                                                           Real,
                                                                                           uEltype <:
                                                                                           Real,
                                                                                           CellBasisT
                                                                                           }
    capacity = length(bases)
    dims = (nvars, ntuple(_ -> n_face_nodes, Val(NDIMS - 1))..., 2 * NDIMS, capacity)

    _surface_flux_values = fill(convert(uEltype, NaN), prod(dims))
    surface_flux_values = unsafe_wrap(Array, pointer(_surface_flux_values), dims)

    return TreeElementBasisContainer{NDIMS, RealT, uEltype, CellBasisT, NDIMS + 2}(bases,
                                                                                   nvars,
                                                                                   num_nodes_in_element,
                                                                                   offsets,
                                                                                   surface_flux_values,
                                                                                   collect(Int,
                                                                                           cell_ids),
                                                                                   _surface_flux_values)
end

@inline Base.real(::TreeElementBasisContainer{NDIMS, RealT}) where {NDIMS,
RealT} = RealT

@inline nelements(elements::TreeElementBasisContainer) = length(elements.bases)

"""
    eachelement(elements::TreeElementBasisContainer)

Return an iterator over the indices that specify the location in relevant data structures
for the elements in `elements`.
In particular, not the elements themselves are returned.
"""
@inline function eachelement(elements::TreeElementBasisContainer)
    Base.OneTo(nelements(elements))
end

num_total_nodes(elements::TreeElementBasisContainer) = elements.offsets[end] ÷
                                                       elements.nvars

function wrap_array(u_ode::AbstractVector, mesh::AbstractMesh, equations,
                    dg::DGFSBP, cache)
    WrappedArray(Val(index_ndims(eltype(cache.elements.bases))), nvariables(equations),
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

# The generic `set_zero!` calls `du[.., element]`, which allocates a view per element.
# Directly zeroing the flat backing vector avoids all allocations.
# `trixi_backend(::WrappedArray)` is always `nothing`, so this covers every case.
function set_zero!(::Nothing, du::WrappedArray, dg::DGFSBP, cache)
    fill!(du.a, zero(eltype(du)))
    return nothing
end

@inline nelements(mesh, dg::DGFSBP, cache) = nelements(cache.elements)
@inline function ndofsglobal(mesh, dg::DGFSBP, cache)
    num_total_nodes(cache.elements)
end

@inline function ndofs(mesh::MeshesDGSEM, dg::DGFSBP, cache)
    num_total_nodes(cache.elements)
end
end # @muladd
