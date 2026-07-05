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
