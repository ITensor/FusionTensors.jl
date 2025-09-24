# This file defines TensorAlgebra interface for a FusionTensor

using LinearAlgebra: mul!

using BlockArrays: Block

using GradedArrays: space_isequal
using TensorAlgebra:
  TensorAlgebra,
  AbstractBlockPermutation,
  BlockedTrivialPermutation,
  BlockedTuple,
  FusionStyle,
  Matricize,
  blockedperm,
  genperm,
  unmatricize

function TensorAlgebra.output_axes(
  ::typeof(contract),
  biperm_dest::AbstractBlockPermutation{2},
  a1::FusionTensor,
  biperm1::AbstractBlockPermutation{2},
  a2::FusionTensor,
  biperm2::AbstractBlockPermutation{2},
  α::Number=one(Bool),
)
  axes_codomain, axes_contracted = blocks(axes(a1)[biperm1])
  axes_contracted2, axes_domain = blocks(axes(a2)[biperm2])
  @assert all(space_isequal.(dual.(axes_contracted), axes_contracted2))
  flat_axes = genperm((axes_codomain..., axes_domain...), Tuple(biperm_dest))
  return FusionTensorAxes(
    tuplemortar((
      flat_axes[begin:length_codomain(biperm_dest)],
      flat_axes[(length_codomain(biperm_dest) + 1):end],
    )),
  )
end

struct FusionTensorFusionStyle <: FusionStyle end

TensorAlgebra.FusionStyle(::Type{<:FusionTensor}) = FusionTensorFusionStyle()

function TensorAlgebra.matricize(
  ::FusionTensorFusionStyle, ft::AbstractArray, biperm::BlockedTrivialPermutation{2}
)
  blocklengths(biperm) == blocklengths(axes(ft)) ||
    throw(ArgumentError("Invalid trivial biperm"))
  return FusionTensor(data_matrix(ft), (codomain_axis(ft),), (domain_axis(ft),))
end

function TensorAlgebra.unmatricize(::FusionTensorFusionStyle, m, blocked_axes)
  return FusionTensor(data_matrix(m), blocked_axes)
end

function TensorAlgebra.permuteblockeddims(
  ft::FusionTensor, biperm::AbstractBlockPermutation
)
  return permutedims(ft, biperm)
end

function TensorAlgebra.permuteblockeddims!(
  a::FusionTensor, b::FusionTensor, biperm::AbstractBlockPermutation
)
  return permutedims!(a, b, biperm)
end

# TODO define custom broadcast rules
function TensorAlgebra.unmatricize_add!(a_dest::FusionTensor, a_dest_mat, invbiperm, α, β)
  a12 = unmatricize(a_dest_mat, axes(a_dest), invbiperm)
  data_matrix(a_dest) .= α .* data_matrix(a12) .+ β .* data_matrix(a_dest)
  return a_dest
end
