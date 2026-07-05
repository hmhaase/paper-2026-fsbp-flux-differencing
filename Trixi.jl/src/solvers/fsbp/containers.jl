"""
    WrappedArray{NDIMS, T, A, WA}

A ragged (jagged) array: unlike a regular `Array`, each element can have its own
node count per dimension. Backed by a single flat vector.

Two ways to index: `u[v, i..., element]` for a single value (`v` = variable,
`i...` = node position per dimension, `element` = which element), or `u[element]`
for a view of that element's whole data, shaped `(nvars, nodes...)`.
"""
struct WrappedArray{NDIMS, T, A, WA}
    nvars::Int
    num_nodes_in_element::Vector{NTuple{NDIMS, Int}} # node count per dimension per element; need not be equal across dimensions
    offsets::Vector{Int} # 0, nvars*prod(nnodes(bases[1])), nvars*prod(nnodes(bases[1])) + nvars*prod(nnodes(bases[2])), ..., nvars*prod(nnodes(bases[1])) + ... + nvars*prod(nnodes(bases[n]))
    a::A

    function WrappedArray(::Val{NDIMS}, nvars, num_nodes_in_element, offsets,
                          a::A) where {NDIMS, T, A <: AbstractVector{T}}
        WA = Trixi.storage_type(a)
        new{NDIMS, T, A, WA}(nvars, num_nodes_in_element, offsets, a)
    end
end

Base.eltype(::WrappedArray{NDIMS, T, A, WA}) where {NDIMS, T, A, WA} = T

trixi_backend(::WrappedArray) = nothing

@inline function convert_index(a::WrappedArray{NDIMS, T, A, WA},
                               (i_vars, i_dims...,
                                element)) where {NDIMS, T,
                                                 A <:
                                                 AbstractVector{T},
                                                 WA
                                                 }
    sizes = a.num_nodes_in_element[element]
    i_vars +
    sum(a.nvars * prod(sizes[1:(d - 1)]) *
        (i_dims[d] - 1) for d in 1:NDIMS) + a.offsets[element]
end

function Base.getindex(a::WrappedArray{NDIMS, T, A, WA},
                       element::Int) where {NDIMS, T, A <: AbstractVector{T}, WA}
    unsafe_wrap(WA, pointer(a.a, a.offsets[element] + 1),
                (a.nvars, a.num_nodes_in_element[element]...))
end

function Base.getindex(a::WrappedArray{NDIMS, T, A, WA},
                       idcs...) where {NDIMS, T,
                                       A <: AbstractVector{T}, WA}
    a.a[convert_index(a, idcs)]
end

function Base.getindex(a::WrappedArray{NDIMS, T, A, WA},
                       ::EllipsisNotation.Ellipsis,
                       element::Int) where {NDIMS, T,
                                            A <: AbstractVector{T}, WA}
    a[element]
end

function Base.setindex!(a::WrappedArray{NDIMS, T, A, WA}, x,
                        idcs...) where {NDIMS, T,
                                        A <: AbstractVector{T}, WA}
    a.a[convert_index(a, idcs)] = x
end
