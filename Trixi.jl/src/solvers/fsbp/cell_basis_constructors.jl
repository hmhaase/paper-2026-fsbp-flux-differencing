# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

abstract type CellBasisConstructor{CellBasisT} end

cell_basis_type(::Type{<:CellBasisConstructor{B}}) where {B} = B
"""
    build_cell_basis(constructor::CellBasisConstructor, pos, dx)

Build the basis for one cell centered at `pos` with width `dx`.
"""
build_cell_basis(::CellBasisConstructor, pos, dx) = error("not implemented")

@inline Base.real(::C) where {RealT, B <: CellBasis{RealT},
C <: CellBasisConstructor{B}} = RealT

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

function build_cell_basis(constructor::LobattoLegendreCellBasisConstructor{RealT},
                          pos, dx) where {RealT <: Real}
    polydeg = constructor.calc_poly_degree(pos, dx)
    return LobattoLegendreCellBasis(polydeg, pos, dx)
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
                                                              domain::Tuple{RealT,
                                                                            RealT}) where {
                                                                                           RealT <:
                                                                                           Real,
                                                                                           CellBasisT <:
                                                                                           CellBasis{RealT}
                                                                                           }
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
    phys_to_ref = nodes_out -> constructor.calc_interpolation_matrix((nodes_out .- pos) ./
                                                                     scale .+ mid_ref)
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

function build_cell_basis(constructor::ReferenceCellBasisConstructor{RealT, CellBasisT},
                          pos, dx) where {RealT <: Real, CellBasisT}
    nodes, weights, D_phys, interp = _map_to_physical(constructor, pos, dx)
    return _build_cell_basis(CellBasisT, nodes, weights, D_phys, interp, pos, dx)
end

struct MixedCellBasisConstructor{RealT, CellBasisT <: CellBasis{RealT},
                                 Constructors <: NamedTuple} <:
       CellBasisConstructor{CellBasisT}
    constructors::Constructors
    selector::Function

    function MixedCellBasisConstructor{RealT, CellBasisT, Constructors}(constructors::Constructors,
                                                                        selector::Function) where {
                                                                                                   RealT <:
                                                                                                   Real,
                                                                                                   CellBasisT <:
                                                                                                   CellBasis{RealT},
                                                                                                   Constructors <:
                                                                                                   NamedTuple
                                                                                                   }
        for name in keys(constructors)
            ctor = getfield(constructors, name)
            @assert ctor isa CellBasisConstructor{<:CellBasis{RealT}} ||
                    error("Constructor '$name' is $(typeof(ctor)), expected CellBasisConstructor{<:CellBasis{$RealT}}")
        end
        new{RealT, CellBasisT, Constructors}(constructors, selector)
    end
end

function MixedCellBasisConstructor(constructors::Constructors,
                                   selector::Function) where {Constructors <:
                                                              NamedTuple}
    CellBasisT = Union{map(ctor -> cell_basis_type(typeof(ctor)),
                           Tuple(constructors))...}
    RealT = real(first(constructors))
    MixedCellBasisConstructor{RealT, CellBasisT, Constructors}(constructors, selector)
end

function build_cell_basis(c::MixedCellBasisConstructor, pos, dx)
    chosen = c.selector(c.constructors, pos, dx)
    return build_cell_basis(chosen, pos, dx)
end

"""
    TensorProductCellBasisConstructor{NDIMS, Ctors, BasisT}

Builds a `TensorProductCellBasis` from one 1D constructor per axis. `pos` and `dx` are
indexed per axis (`pos[d]`, `dx[d]`), so cells need not be square.
"""
struct TensorProductCellBasisConstructor{NDIMS, Ctors <: Tuple,
                                         BasisT <: CellBasis} <:
       CellBasisConstructor{BasisT}
    constructors::Ctors
end

function TensorProductCellBasisConstructor(constructors::Tuple)
    RealT = real(first(constructors))
    basis_types = map(c -> cell_basis_type(typeof(c)), constructors)
    # Varying the basis across elements needs a `MixedCellBasisConstructor` *of* tensor
    # constructors, not a tensor product of mixed ones. #TODOFSBP: support that.
    all(isconcretetype, basis_types) ||
        error("tensor-product cell bases need one concrete basis type per axis")
    BasisT = TensorProductCellBasis{length(constructors), RealT, Tuple{basis_types...}}
    return TensorProductCellBasisConstructor{length(constructors), typeof(constructors),
                                             BasisT}(constructors)
end

function build_cell_basis(c::TensorProductCellBasisConstructor{NDIMS}, pos,
                          dx) where {NDIMS}
    bases = ntuple(d -> build_cell_basis(c.constructors[d], pos[d], dx[d]), Val(NDIMS))
    return TensorProductCellBasis(bases)
end

# Wrap a 1D constructor into an isotropic `NDIMS`-axis tensor product; leave a tensor one.
as_tensor(c::TensorProductCellBasisConstructor, ::Val) = c
function as_tensor(c::CellBasisConstructor, ::Val{NDIMS}) where {NDIMS}
    return TensorProductCellBasisConstructor(ntuple(_ -> c, Val(NDIMS)))
end
end # @muladd
