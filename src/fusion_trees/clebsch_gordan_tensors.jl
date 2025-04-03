# This file defines Clebsch-Gordan tensors
# one tensor is defined from 3 simple objects s1, s2 and s3
# and contains the coefficients fusing s1 ⊗ s2 -> s3

using HalfIntegers: half
using WignerSymbols: clebschgordan

using GradedArrays: dual
using GradedArrays.SymmetrySectors:
  AbelianStyle,
  AbstractSector,
  NotAbelianStyle,
  SymmetryStyle,
  O2,
  SU,
  istrivial,
  quantum_dimension,
  sector_label,
  trivial,
  zero_odd
using TensorAlgebra: contract
using TensorProducts: ⊗

function symbol_1j(s::AbstractSector)
  cgt = clebsch_gordan_tensor(s, dual(s), trivial(s), 1)
  return sqrt(quantum_dimension(s)) * cgt[:, :, 1]
end

function clebsch_gordan_tensor(
  s1::AbstractSector,
  s2::AbstractSector,
  s3::AbstractSector,
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

function clebsch_gordan_tensor(s1::S, s2::S, s3::S, outer_mult_index::Int) where {S}
  return clebsch_gordan_tensor(SymmetryStyle(S), s1, s2, s3, outer_mult_index)
end

function clebsch_gordan_tensor(
  ::AbelianStyle, s1::S, s2::S, s3::S, outer_mult_index::Int
) where {S}
  @assert outer_mult_index == 1
  return s1 ⊗ s2 == s3 ? ones((1, 1, 1)) : zeros((1, 1, 1))
end

function clebsch_gordan_tensor(::NotAbelianStyle, s1::O2, s2::O2, s3::O2, ::Int)
  return clebsch_gordan_tensor(s1, s2, s3)  # no outer multiplicity
end

function clebsch_gordan_tensor(s1::O2, s2::O2, s3::O2)
  d1 = quantum_dimension(s1)
  d2 = quantum_dimension(s2)
  d3 = quantum_dimension(s3)
  cgt = zeros((d1, d2, d3))
  s3 ∉ blocklabels(s1 ⊗ s2) && return cgt

  # adapted from TensorKit
  l1 = sector_label(s1)
  l2 = sector_label(s2)
  l3 = sector_label(s3)
  if l3 <= 0  # 0even or 0odd
    if l1 <= 0 && l2 <= 0
      cgt[1, 1, 1, 1] = 1.0
    else
      if istrivial(s3)
        cgt[1, 2, 1, 1] = 1.0 / sqrt(2)
        cgt[2, 1, 1, 1] = 1.0 / sqrt(2)
      else
        cgt[1, 2, 1, 1] = 1.0 / sqrt(2)
        cgt[2, 1, 1, 1] = -1.0 / sqrt(2)
      end
    end
  elseif l1 <= 0  # 0even or 0odd
    cgt[1, 1, 1, 1] = 1.0
    cgt[1, 2, 2, 1] = s1 == zero_odd(O2) ? -1.0 : 1.0
  elseif l2 == 0
    cgt[1, 1, 1, 1] = 1.0
    cgt[2, 1, 2, 1] = s2 == zero_odd(O2) ? -1.0 : 1.0
  elseif l3 == l1 + l2
    cgt[1, 1, 1, 1] = 1.0
    cgt[2, 2, 2, 1] = 1.0
  elseif l3 == l1 - l2
    cgt[1, 2, 1, 1] = 1.0
    cgt[2, 1, 2, 1] = 1.0
  elseif l3 == l2 - l1
    cgt[2, 1, 1, 1] = 1.0
    cgt[1, 2, 2, 1] = 1.0
  end
  return cgt
end

function clebsch_gordan_tensor(::NotAbelianStyle, s1::SU{2}, s2::SU{2}, s3::SU{2}, ::Int)
  return clebsch_gordan_tensor(s1, s2, s3)  # no outer multiplicity
end

function clebsch_gordan_tensor(s1::SU{2}, s2::SU{2}, s3::SU{2})
  d1 = quantum_dimension(s1)
  d2 = quantum_dimension(s2)
  d3 = quantum_dimension(s3)
  j1 = half(d1 - 1)
  j2 = half(d2 - 1)
  j3 = half(d3 - 1)
  cgtensor = Array{Float64,3}(undef, (d1, d2, d3))
  for (i, j, k) in Iterators.product(1:d1, 1:d2, 1:d3)
    m1 = j1 - i + 1
    m2 = j2 - j + 1
    m3 = j3 - k + 1
    cgtensor[i, j, k] = clebschgordan(j1, m1, j2, m2, j3, m3)
  end
  return cgtensor
end

function clebsch_gordan_tensor(
  ::NotAbelianStyle, s1::SU{3}, s2::SU{3}, s3::SU{3}, outer_mult_index::Int
)
  d1 = quantum_dimension(s1)
  d2 = quantum_dimension(s2)
  d3 = quantum_dimension(s3)
  cgtensor = zeros(d1, d2, d3)
  # dummy
  return cgtensor
end
