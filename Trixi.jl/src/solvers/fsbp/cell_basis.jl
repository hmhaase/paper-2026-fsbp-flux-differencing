"Marker for the face at the negative end of an axis (left in 1D)."
struct Negative end
"Marker for the face at the positive end of an axis (right in 1D)."
struct Positive end

"""
    CellBasis{RealT}

Base type for a basis on one element. Must implement:
- real(::CellBasis)
- ndofs_per_element(::CellBasis)
- eachnode(::CellBasis)
- index_ndims(::CellBasis)
"""
abstract type CellBasis{RealT} end

"""
    LineBasis{RealT} <: CellBasis{RealT}

A 1D SBP operator (one node set + its matrices/boundary operators). In addition to the
`CellBasis` methods, must implement:
- get_nodes(::LineBasis)
- nnodes(::LineBasis)
- quadrature_weights(::LineBasis)
- interpolation_matrix(::LineBasis, nodes_out)
- boundary_value!, boundary_coordinate

Fields:
- boundary_lift    # M⁻¹Rᵀ, [nnodes, 2] (Lobatto: R = [e_1, e_N]ᵀ)
- derivative_split # 2D - M⁻¹B, skew-symmetric split form (flux differencing)
- derivative_dhat  # weak form, dhat[j,n] = -D'[j,n] w[n] / w[j]
"""
abstract type LineBasis{RealT} <: CellBasis{RealT} end

@inline index_ndims(basis::CellBasis) = index_ndims(typeof(basis))
@inline index_ndims(::Type{<:LineBasis}) = 1

"""
    LobattoLikeCellBasis <: LineBasis

Basis including boundary nodes with quadrature_weights similar to a Gauss-Lobatto quadrature rule.
"""
struct LobattoLikeCellBasis{RealT <: Real} <: LineBasis{RealT}
    _nodes::Vector{RealT}
    _weights::Vector{RealT}
    # _inverse_weights::Vector{RealT}

    # _inverse_vandermonde_legendre::Matrix{RealT}
    "Matrix L such that \$L^T M u = [u_L, u_R]^T\$, that is \$L = M⁻¹ [e_1, e_N]\$"
    boundary_lift::Matrix{RealT} # lhat

    # _derivative_matrix::Matrix{RealT} # strong form derivative matrix
    derivative_split::Matrix{RealT} # 2D - M⁻¹B, skew-symmetric split form
    # _derivative_split_transpose::Matrix{RealT} # transpose of `derivative_split`
    derivative_dhat::Matrix{RealT} # weak form matrix "dhat"

    """
        calc_interpolation_matrix(nodes_out)

    Return the interpolation matrix from the basis nodes to `nodes_out`.
    `nodes_out` are in the same coordinate space as the basis nodes (physical space).
    The basis nodes are captured in the callable, it takes only the target nodes.
    Returns a matrix `V` of size `(length(nodes_out), nnodes)` such that
    `u_target = V * u_basis` interpolates from basis node values to target node values.
    """
    calc_interpolation_matrix::Any
end

get_nodes(basis::LobattoLikeCellBasis) = basis._nodes
eachnode(basis::LobattoLikeCellBasis) = Base.OneTo(nnodes(basis))
nnodes(basis::LobattoLikeCellBasis) = length(basis._nodes)
ndofs_per_element(basis::LobattoLikeCellBasis) = nnodes(basis)
quadrature_weights(basis::LobattoLikeCellBasis) = basis._weights

# TODO: Consider specifying a formal description of where the new nodes should be
# for example n gauss lobatto nodes within the domain of the basis and then caching
# the vandermonde matrix
#
# Currently required for error calculation and plotting
function interpolation_matrix(basis::LobattoLikeCellBasis, nodes_out)
    basis.calc_interpolation_matrix(nodes_out)
end

@inline Base.real(::LobattoLikeCellBasis{RealT}) where {RealT <: Real} = RealT

@inline boundary_coordinate(basis::LobattoLikeCellBasis, ::Negative) = basis._nodes[1]
@inline boundary_coordinate(basis::LobattoLikeCellBasis, ::Positive) = basis._nodes[end]

function LobattoLikeCellBasis(nodes, weights, derivative_matrix, calc_interpolation_matrix)
    RealT = eltype(nodes)
    @assert RealT <: Real
    boundary_lift = zeros(RealT, length(nodes), 2)
    boundary_lift[1, 1] = 1 / weights[1]
    boundary_lift[end, end] = 1 / weights[end]
    # 2D - M⁻¹B
    #
    # M⁻¹ = diagm(1 ./ weights)
    # B = diagm(-1, 0, ..., 0, 1)
    # => M⁻¹B = diagm(-1/w_1, 0, ..., 0, 1/w_end)
    derivative_split = 2 * derivative_matrix
    derivative_split[1, 1] -= -1 / weights[1]
    derivative_split[end, end] -= 1 / weights[end]

    # Weak form matrix: dhat[j,n] = -D'[j,n] * w[n] / w[j]
    derivative_dhat = -derivative_matrix' .* weights' ./ weights

    LobattoLikeCellBasis{RealT}(nodes, weights, boundary_lift, derivative_split,
                                derivative_dhat, calc_interpolation_matrix)
end

"""
    LobattoLegendreCellBasis

Construct the standard Lobatto Legendre Basis with Gauss-Lobatto quadrature.
"""
function LobattoLegendreCellBasis(polydeg::Int, pos::RealT, dx::RealT) where {RealT <: Real}
    nnodes_ = polydeg + 1

    nodes, weights = gauss_lobatto_nodes_weights(nnodes_, RealT)
    J = dx / 2
    nodes = J .* nodes .+ pos
    weights = abs(J) .* weights
    # inverse_weights = inv.(weights)

    # _, inverse_vandermonde_legendre = vandermonde_legendre(nodes, RealT)

    boundary_lift = zeros(RealT, nnodes_, 2)
    boundary_lift[:, 1] = calc_Lhat(calc_L(J * -one(RealT) + pos, nodes, weights),
                                    weights)
    boundary_lift[:, 2] = calc_Lhat(calc_L(J * one(RealT) + pos, nodes, weights),
                                    weights)

    derivative_matrix = polynomial_derivative_matrix(nodes)
    derivative_split = calc_Dsplit(derivative_matrix, weights)
    # derivative_split_transpose = Matrix(derivative_split')
    derivative_dhat = calc_Dhat(derivative_matrix, weights)

    calc_interp = nodes_out -> polynomial_interpolation_matrix(nodes, nodes_out)

    LobattoLikeCellBasis{RealT}(nodes, weights,
                                # inverse_weights,
                                # inverse_vandermonde_legendre,
                                boundary_lift,
                                # derivative_matrix,
                                derivative_split,
                                # derivative_split_transpose,
                                derivative_dhat,
                                calc_interp)
end

"""
    GeneralSBPCellBasis <: LineBasis

Basis for general SBP operators where nodes need not include element boundaries.
The boundary interpolation/extrapolation operator R is stored as `boundary_interpolation`
(shape `(2, nnodes)`, row 1 = left/−x, row 2 = right/+x) and maps node values to boundary
values.  `boundary_lift = M⁻¹Rᵀ` (shape `(nnodes, 2)`) is pre-computed for the surface
integral hot path.
"""
struct GeneralSBPCellBasis{RealT <: Real} <: LineBasis{RealT}
    _nodes::Vector{RealT}
    _weights::Vector{RealT}
    "M⁻¹Rᵀ, size (nnodes, 2); same field name and layout as LobattoLikeCellBasis.boundary_lift"
    boundary_lift::Matrix{RealT}
    "Boundary evaluation operator R, size (2, nnodes); row 1 = left (−x), row 2 = right (+x)"
    boundary_interpolation::Matrix{RealT}
    "Physical coordinates [x_left, x_right] of the element boundaries"
    boundary_nodes::SVector{2, RealT}
    derivative_split::Matrix{RealT}
    derivative_dhat::Matrix{RealT}
    calc_interpolation_matrix::Any
end

get_nodes(basis::GeneralSBPCellBasis) = basis._nodes
eachnode(basis::GeneralSBPCellBasis) = Base.OneTo(nnodes(basis))
nnodes(basis::GeneralSBPCellBasis) = length(basis._nodes)
ndofs_per_element(basis::GeneralSBPCellBasis) = nnodes(basis)
quadrature_weights(basis::GeneralSBPCellBasis) = basis._weights

function interpolation_matrix(basis::GeneralSBPCellBasis, nodes_out)
    basis.calc_interpolation_matrix(nodes_out)
end

@inline Base.real(::GeneralSBPCellBasis{RealT}) where {RealT <: Real} = RealT

@inline boundary_coordinate(basis::GeneralSBPCellBasis, ::Negative) = basis.boundary_nodes[1]
@inline boundary_coordinate(basis::GeneralSBPCellBasis, ::Positive) = basis.boundary_nodes[2]

function GeneralSBPCellBasis(nodes, boundary_nodes, weights, D, calc_interpolation_matrix)
    RealT = eltype(nodes)
    n = length(nodes)
    @assert size(D) == (n, n)

    # Derive R by evaluating the interpolation operator at the physical boundary points
    boundary_interpolation = calc_interpolation_matrix(boundary_nodes)
    @assert size(boundary_interpolation) == (2, n)

    # boundary_lift = M⁻¹Rᵀ:  lift[i,j] = R[j,i] / w[i]
    boundary_lift = Matrix{RealT}(undef, n, 2)
    for j in 1:2, i in 1:n
        boundary_lift[i, j] = boundary_interpolation[j, i] / weights[i]
    end

    # derivative_split = 2D − M⁻¹B,  B = Rᵀ diag(−1, 1) R
    # [-1,1] .* boundary_interpolation scales row 1 by -1 and row 2 by +1 (diag(-1,1)*R)
    # then (n×2)*(2×n) gives B
    B = boundary_interpolation' * ([-one(RealT), one(RealT)] .* boundary_interpolation)
    derivative_split = 2 * D - (1 ./ weights) .* B

    # derivative_dhat = −Dᵀ .* wᵀ ./ w  (weak form matrix)
    derivative_dhat = -D' .* weights' ./ weights

    GeneralSBPCellBasis{RealT}(Vector{RealT}(nodes), Vector{RealT}(weights),
                               Matrix{RealT}(boundary_lift),
                               Matrix{RealT}(boundary_interpolation),
                               SVector{2, RealT}(boundary_nodes),
                               Matrix{RealT}(derivative_split),
                               Matrix{RealT}(derivative_dhat),
                               calc_interpolation_matrix)
end

"""
    TensorProductCellBasis{NDIMS, RealT, B} <: CellBasis{RealT}

Tensor product of `NDIMS` `LineBasis`es (one per axis; types may differ).
"""
struct TensorProductCellBasis{NDIMS, RealT,
                              B <: Tuple{Vararg{LineBasis{RealT}}}} <: CellBasis{RealT}
    bases::B
end

function TensorProductCellBasis(bases::Tuple)
    RealT = real(first(bases))
    return TensorProductCellBasis{length(bases), RealT, typeof(bases)}(bases)
end

@inline index_ndims(::Type{<:TensorProductCellBasis{NDIMS}}) where {NDIMS} = NDIMS
@inline Base.real(::TensorProductCellBasis{NDIMS, RealT}) where {NDIMS, RealT} = RealT
@inline ndofs_per_element(basis::TensorProductCellBasis) = prod(nnodes, basis.bases)
@inline eachnode(basis::TensorProductCellBasis) = CartesianIndices(map(nnodes, basis.bases))
