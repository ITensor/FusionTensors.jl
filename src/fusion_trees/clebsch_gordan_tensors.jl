# This file defines Clebsch-Gordan tensors
# one tensor is defined from 3 simple objects s1, s2 and s3
# and contains the coefficients fusing s1 ⊗ s2 -> s3

using HalfIntegers: half
using WignerSymbols: clebschgordan

using GradedArrays:
    AbelianStyle,
    NotAbelianStyle,
    O2,
    SU2,
    SectorRange,
    SymmetryStyle,
    TrivialSector,
    dual,
    istrivial,
    quantum_dimension,
    sector_label,
    sectors,
    trivial,
    zero_odd
using TensorAlgebra: contract
using TensorProducts: ⊗
import TensorKitSectors as TKS

function symbol_1j(s::SectorRange)
    cgt = clebsch_gordan_tensor(s, dual(s), trivial(s), 1)
    return sqrt(quantum_dimension(s)) * cgt[:, :, 1]
end

function clebsch_gordan_tensor(
        s1::SectorRange,
        s2::SectorRange,
        s3::SectorRange,
        arrow1::Bool,
        arrow2::Bool,
        inner_mult_index::Int,
    )
    cgt = clebsch_gordan_tensor(s1, s2, s3, inner_mult_index)
    if arrow1
        flip1 = symbol_1j(s1)
        cgt = contract((1, 2, 3), flip1, (4, 1), cgt, (4, 2, 3))
    end
    if arrow2
        flip2 = symbol_1j(s2)
        cgt = contract((1, 2, 3), flip2, (4, 2), cgt, (1, 4, 3))
    end
    return cgt
end

function clebsch_gordan_tensor(s1::S, s2::S, s3::S, outer_mult_index::Int = 1) where {S}
    CGC = TKS.fusiontensor(GradedArrays.label.((s1, s2, s3))...)
    outer_mult_index ∈ axes(CGC, 4) || throw(ArgumentError("invalid outer multiplicity index"))
    if TKS.FusionStyle(S) === TKS.GenericFusion()
        # TODO: do we want a view here?
        return CGC[:, :, :, outer_mult_index]
    else
        return dropdims(CGC; dims = 4)
    end
end

# TODO: remove once TensorKitSectors fixes this
function clebsch_gordan_tensor(s1::TrivialSector, s2::TrivialSector, s3::TrivialSector, outer_mult_index::Int = 1)
    outer_mult_index == 1 || throw(ArgumentError("invalid outer multiplicity index"))
    return fill(1, (1, 1, 1))
end
