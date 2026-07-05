"Marker for the face at the negative end of an axis (left in 1D)."
struct Negative end
"Marker for the face at the positive end of an axis (right in 1D)."
struct Positive end

"""
    CellBasis

Base type for all cell local bases. Must implement:

Methods:
- get_nodes(::CellBasis)
- eachnode(::CellBasis)
- nnodes(::CellBasis)
- quadrature_weights(::CellBasis)
- interpolation_matrix(::CellBasis, nodes_out)
- real(::CellBasis)

Attributes:
- derivative_split
  2D - M⁻¹B for strong derivative matrix D
- boundary_lift
  Matrix L such that \$L^T M u = [u_L, u_R]T\$, that is \$L = M⁻¹ [e_1, e_N]\$
- derivative_dhat
  Weak form matrix dhat such that (dhat * f)_j approximates the weak derivative
  at node j: dhat[j,n] = -D'[j,n] * w[n] / w[j] where D is the strong-form
  derivative matrix and w are the quadrature weights.
"""
abstract type CellBasis{RealT} end

"""
    LobattoLikeCellBasis <: CellBasis

Basis including boundary nodes with quadrature_weights similar to a Gauss-Lobatto quadrature rule.
"""
struct LobattoLikeCellBasis{RealT <: Real} <: CellBasis{RealT}
    _nodes::Vector{RealT}
    _weights::Vector{RealT}
    # _inverse_weights::Vector{RealT}

    # _inverse_vandermonde_legendre::Matrix{RealT}
    "Matrix L such that \$L^T M u = [u_L, u_R]T\$, that is \$L = M⁻¹ [e_1, e_N]\$"
    boundary_lift::Matrix{RealT} # lhat

    # _derivative_matrix::Matrix{RealT} # strong form derivative matrix
    derivative_split::Matrix{RealT} # strong form derivative matrix minus boundary terms
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
