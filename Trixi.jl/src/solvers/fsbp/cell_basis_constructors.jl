abstract type CellBasisConstructor{CellBasisT} end

cell_basis_type(::Type{<:CellBasisConstructor{B}}) where {B} = B
"""
    init_local_basis!(constructor::CellBasisConstructor, basis_ref, args...)

Initalize an allocated basis for a given cell.
    basis_ref[] = ...
"""
init_local_basis!(::CellBasisConstructor, basis_ref, args...) = error("not implemented")

@inline Base.real(::C) where {RealT, B <: CellBasis{RealT}, C <: CellBasisConstructor{B}} = RealT

struct LobattoLegendreCellBasisConstructor{RealT} <:
       CellBasisConstructor{LobattoLikeCellBasis{RealT}}
    """
        calc_poly_degree(pos, dx)

    Calculate the degree of the polynomial to be constructed at cell with position `pos` and length `dx`.
    """
    calc_poly_degree::Function

    function LobattoLegendreCellBasisConstructor{RealT}(calc_poly_degree) where {RealT <:
                                                                                 Real}
        new(calc_poly_degree)
    end
end

function init_local_basis!(constructor::LobattoLegendreCellBasisConstructor{RealT},
                           basis_ref, pos, dx) where {RealT <: Real}
    polydeg = constructor.calc_poly_degree(pos, dx)
    basis_ref[] = LobattoLegendreCellBasis(polydeg, pos, dx)
end

"""
    ReferenceCellBasisConstructor{RealT, CellBasisT}

Wraps one fixed, precomputed reference-domain SBP operator (`D`, `x`, `w` on `domain`)
and affinely maps it to each cell, rather than recomputing an operator per cell like
`LobattoLegendreCellBasisConstructor` does.
"""
struct ReferenceCellBasisConstructor{RealT, CellBasisT <: CellBasis{RealT}} <:
       CellBasisConstructor{CellBasisT}
    D::Matrix{RealT}
    x::Vector{RealT}
    w::Vector{RealT}
    calc_interpolation_matrix::Any
    domain::Tuple{RealT, RealT}

    function ReferenceCellBasisConstructor{RealT, CellBasisT}(D, x, w,
                                                              calc_interpolation_matrix,
                                                              domain::Tuple{RealT, RealT}) where {RealT <:
                                                                                                   Real,
                                                                                                   CellBasisT <:
                                                                                                   CellBasis{RealT}}
        @assert size(D, 1) == size(D, 2) == length(x) == length(w)
        @assert domain[1] < domain[2]
        new{RealT, CellBasisT}(Matrix{RealT}(D), Vector{RealT}(x), Vector{RealT}(w),
                               calc_interpolation_matrix, domain)
    end
end

# Shared helper: map reference-domain SBP operators to physical space for an element at
# `pos` with width `dx`.  Works for any constructor with fields D, x, w,
# calc_interpolation_matrix, domain.
function _map_to_physical(constructor, pos, dx)
    a, b = constructor.domain
    scale = dx / (b - a)
    mid_ref = (a + b) / 2
    nodes = scale .* (constructor.x .- mid_ref) .+ pos
    weights = scale .* constructor.w
    D_phys = constructor.D ./ scale
    phys_to_ref = nodes_out -> constructor.calc_interpolation_matrix(
        (nodes_out .- pos) ./ scale .+ mid_ref)
    return nodes, weights, D_phys, phys_to_ref
end

# The one place the two `CellBasis` families differ: how to build the final basis
# from the mapped (nodes, weights, D_phys, interp) tuple.
function _build_cell_basis(::Type{<:LobattoLikeCellBasis},
                           nodes, weights, D_phys, interp, pos, dx)
    LobattoLikeCellBasis(nodes, weights, D_phys, interp)
end

function _build_cell_basis(::Type{<:GeneralSBPCellBasis},
                           nodes, weights, D_phys, interp, pos, dx)
    RealT = eltype(nodes)
    J = dx / 2
    boundary_nodes = SVector{2, RealT}(pos - J, pos + J)
    GeneralSBPCellBasis(nodes, boundary_nodes, weights, D_phys, interp)
end

function init_local_basis!(constructor::ReferenceCellBasisConstructor{RealT, CellBasisT},
                           basis_ref, pos, dx) where {RealT <: Real, CellBasisT}
    nodes, weights, D_phys, interp = _map_to_physical(constructor, pos, dx)
    basis_ref[] = _build_cell_basis(CellBasisT, nodes, weights, D_phys, interp, pos, dx)
end
