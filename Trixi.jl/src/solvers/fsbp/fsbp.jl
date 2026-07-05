include("containers.jl")

abstract type FSBPBasis end

include("cell_basis.jl")
include("cell_basis_constructors.jl")

"""
    PerCellBasis{CellBasisT}

Each cell of the mesh has its own CellBasis.
Marker type that holds no data itself.
Everything is delegated to bases for individual cells/elements.
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

- analysis_polydeg == nothing => no interpolation, compares numerical solution to exact solution at basis nodes
- analysis_polydeg == n::Int => projects exact and numerical solution to polynomial space with degree n and compares these projections
"""
function SolutionAnalyzer(basis::PerCellBasis; analysis_polydeg = nothing)
    RealT = real(basis)
    return PerCellAnalyzer{RealT}(analysis_polydeg)
end

include("calc_volume_integral.jl")
