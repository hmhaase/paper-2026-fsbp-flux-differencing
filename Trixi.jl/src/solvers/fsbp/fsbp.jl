# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

include("containers.jl")

abstract type FSBPBasis end

include("cell_basis.jl")
include("cell_basis_constructors.jl")

"""
    PerCellBasis{CellBasisConstructorT}

Each cell of the mesh has its own CellBasis. Holds only the `local_basis_constructor`
that builds the basis of each cell; everything else is delegated to the bases of the
individual cells/elements.
"""
struct PerCellBasis{CellBasisConstructorT} <:
       FSBPBasis
    local_basis_constructor::CellBasisConstructorT
end

function PerCellBasis(local_basis_constructor::CellBasisConstructorT) where
         {
          CellBasisConstructorT <:
          CellBasisConstructor
          }
    PerCellBasis{CellBasisConstructorT}(local_basis_constructor)
end

const DGFSBP = DG{Basis} where {Basis <: FSBPBasis}

function DGFSBP(cell_basis_constructor;
                basis = PerCellBasis(cell_basis_constructor),
                surface_flux = flux_central,
                surface_integral = SurfaceIntegralWeakForm(surface_flux),
                volume_integral = VolumeIntegralFluxDifferencing(),
                mortar = nothing)
    return DG{typeof(basis), typeof(mortar), typeof(surface_integral),
              typeof(volume_integral)}(basis, mortar, surface_integral, volume_integral)
end

@inline Base.real(::PerCellBasis{CellBasisConstructorT}) where {
RealT,
CellBasisT <: CellBasis{RealT},
CellBasisConstructorT <: CellBasisConstructor{CellBasisT}
} = RealT

struct PerCellAnalyzer{RealT <: Real} <:
       SolutionAnalyzer{RealT}
    analysis_polydeg::Any
end

"""
    SolutionAnalyzer(basis::PerCellBasis; analysis_polydeg = nothing)

SolutionAnalyzer for spatially varying bases.

- `analysis_polydeg == nothing`: no interpolation; compares the numerical solution to the
  exact solution at the basis nodes.
- `analysis_polydeg == n::Int`: evaluates the exact solution and interpolates the numerical
  solution at n + 1 Gauss-Lobatto nodes per element and compares them there. 1D only; on a
  2D mesh this errors.
"""
function SolutionAnalyzer(basis::PerCellBasis; analysis_polydeg = nothing)
    RealT = real(basis)
    return PerCellAnalyzer{RealT}(analysis_polydeg)
end

include("calc_volume_integral.jl")
end # @muladd
