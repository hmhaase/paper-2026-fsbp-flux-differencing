# Cell-basis constructors for `DGFSBP`; entry points `fsbp_cell_operator`, `fsbp_space_dim`.

using LinearAlgebra
using GenericLinearAlgebra   # BigFloat SVD, needed by `pinv` below
using Trixi

include(joinpath(@__DIR__, "fsbp_ops_exp_physical_closed.jl"))
include(joinpath(@__DIR__, "fsbp_ops_exp_exp2_physical_closed.jl"))
include(joinpath(@__DIR__, "fsbp_ops_trig_physical_closed.jl"))

# Reference basis on a cell of half-width `s`. The spaces are translation invariant, so
# `s` alone fixes the reference space, and one operator per cell width serves every cell.
_exp_basis(n_exp) = s -> vcat([ξ -> one(ξ), ξ -> ξ],
                              [(ξ -> exp(k * s * ξ)) for k in 1:n_exp])
_trig_basis(ω) = s -> [ξ -> one(ξ), ξ -> ξ, ξ -> sin(ω * s * ξ), ξ -> cos(ω * s * ξ)]

# Supported spaces by name:
#   E2 = span{1, x, e^x}
#   E* = span{1, x, e^x, e^{2x}}
#   T* = span{1, x, sin ωx, cos ωx}, with ω = PHYSICAL_TRIG_OMEGA

# Key each table by its entries' own `h`, so a lookup does not depend on which level
# range the table holds.
function _by_width(ops)
    table = Dict{Int, eltype(ops)}()
    for op in ops
        key = round(Int, -log2(Float64(op.h)))
        haskey(table, key) &&
            error("two operators for cell width h = $(Float64(op.h))")
        table[key] = op
    end
    return table
end

const SPACES = Dict("E2" => (_by_width(PHYSICAL_EXP_OPS_CLOSED), _exp_basis(1)),
                    "E*" => (_by_width(PHYSICAL_EXP_EXP2_OPS_CLOSED), _exp_basis(2)),
                    "T*" => (_by_width(PHYSICAL_TRIG_OPS_CLOSED),
                             _trig_basis(PHYSICAL_TRIG_OMEGA)))

function _space_entry(space::String)
    haskey(SPACES, space) ||
        error("unknown FSBP space $(repr(space)); known: $(sort(collect(keys(SPACES))))")
    return SPACES[space]
end

# Mixed spaces: components named left to right of a split coordinate, each a space in
# `SPACES` or "P<p>", degree-p polynomials on Lobatto-Legendre nodes.
const MIXED_SPACES = Dict("P3|E2" => ("P3", "E2"))

function _poly_degree(name::String)
    m = match(r"^P(\d+)$", name)
    return isnothing(m) ? nothing : parse(Int, m[1])
end

"""
    fsbp_space_dim(space::String)

Dimension of the function space `space` names, e.g. 3 for "E2" and p + 1 for "P<p>".
A mixed space returns one dimension per component, in the order its name lists them.
"""
function fsbp_space_dim(space::String)
    haskey(MIXED_SPACES, space) &&
        return [_component_dim(name) for name in MIXED_SPACES[space]]
    return _component_dim(space)
end

function _component_dim(name::String)
    p = _poly_degree(name)
    isnothing(p) || return p + 1
    _, basis = _space_entry(name)
    return length(basis(one(Float64)))
end

# A width the table does not hold is an error, never a silently mismatched operator.
function _table_entry(space, table, h)
    k = -log2(h)
    key = round(Int, k)
    isapprox(k, key; atol = 1e-9) ||
        error("no $space FSBP operator for cell width h = $h: " *
              "implied key $k is not an integer")
    haskey(table, key) ||
        error("$space FSBP operator table has no entry for cell width h = $h (key $key); " *
              "available keys: $(sort(collect(keys(table))))")
    return table[key]
end

"""
    fsbp_cell_operator(space::String, cell_width::Real; RealT = Float64, split_at = nothing)

Cell-basis constructor for `space`, ready to pass to `DGFSBP`. The operator is selected
by `cell_width` alone, so every cell of the mesh must have that width; it is an error if
no operator is tabulated for it. `"P<p>"` ignores `cell_width`.

Every operator has a node on each cell boundary.

For a mixed space, elements centred below `split_at` get the first component, the rest
the second.

The tables are 256-bit BigFloat literals, rounded once to `RealT`; raising `setprecision`
past 256 bits does not refine them.
"""
function fsbp_cell_operator(space::String, cell_width::Real;
                            RealT::Type = Float64, split_at = nothing)
    q = _poly_degree(space)
    isnothing(q) || return _polynomial_operator(q, RealT)
    haskey(MIXED_SPACES, space) || return _tabulated_operator(space, cell_width, RealT)
    isnothing(split_at) &&
        error("the mixed space $(repr(space)) needs `split_at`, the coordinate where its " *
              "two bases meet")
    left, right = MIXED_SPACES[space]
    constructors = (; left = _component_operator(left, cell_width, RealT),
                    right = _component_operator(right, cell_width, RealT))
    return MixedCellBasisConstructor(constructors,
                                     (c, pos, dx) -> pos < split_at ? c.left : c.right)
end

function _component_operator(name::String, cell_width::Real, RealT::Type)
    p = _poly_degree(name)
    isnothing(p) || return _polynomial_operator(p, RealT)
    return _tabulated_operator(name, cell_width, RealT)
end

function _polynomial_operator(p::Int, RealT::Type)
    return LobattoLegendreCellBasisConstructor{RealT}((pos, dx) -> p)
end

function _tabulated_operator(space::String, cell_width::Real, RealT::Type)
    table, basis = _space_entry(space)
    op = _table_entry(space, table, cell_width)
    return _reference_operator(op, basis, cell_width, RealT,
                               Trixi.LobattoLikeCellBasis{RealT})
end

function _reference_operator(op, basis, cell_width::Real, RealT::Type, CellBasisT::Type)
    # BigFloat whatever `RealT`: on small cells `V` is too ill-conditioned for Float64.
    funcs = basis(big(cell_width) / 2)
    n_funcs = length(funcs)
    n_nodes = length(op.x)
    V = [funcs[j](big(op.x[i])) for i in 1:n_nodes, j in 1:n_funcs]
    # More nodes than basis functions, so this is a least-squares fit in the space, not an
    # interpolant; Trixi uses it only for error analysis and plotting.
    V_pinv = pinv(V)
    calc_interp = nodes_ref -> RealT.([funcs[j](big(nodes_ref[i]))
                                       for i in 1:length(nodes_ref), j in 1:n_funcs] *
                                      V_pinv)

    return ReferenceCellBasisConstructor{RealT, CellBasisT}(RealT.(op.D), RealT.(op.x),
                                                            RealT.(op.w), calc_interp,
                                                            (RealT(op.domain[1]),
                                                             RealT(op.domain[2])))
end
