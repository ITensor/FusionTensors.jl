# This file defines TensorAlgebra interface for a FusionTensor

using LinearAlgebra: mul!

using BlockArrays: Block

using TensorAlgebra: BlockedPermutation, Matricize, TensorAlgebra

# TODO how to deal with inner contraction = no ouput axis?
# => currently biperm_dest is a BlockedPermutation{0}, change this
function TensorAlgebra.allocate_output(
  ::typeof(contract),
  biperm_dest::BlockedPermutation{2},
  a1::FusionTensor,
  biperm1::BlockedPermutation{2},
  a2::FusionTensor,
  biperm2::BlockedPermutation{2},
  α::Number=true,
)
  axes_dest = (
    map(i -> axes(a1)[i], first(blocks(biperm1))),
    map(i -> axes(a2)[i], last(blocks(biperm2))),
  )
  return similar(a1, promote_type(eltype(a1), eltype(a2), typeof(α)), axes_dest)
end

# TBD do really I need to define these as I cannot use them in contract! and has to redefine it?
#TensorAlgebra.fusedims(ft::FusionTensor, perm::BlockedPermutation{2}) = permutedims(ft, perm)
#function TensorAlgebra.splitdims(ft1::FusionTensor, ft2::FusionTensor, blockedperm::BlockedPermutation)
#function TensorAlgebra.splitdims!(ft1::FusionTensor, ft2::FusionTensor, blockedperm::BlockedPermutation)

# I cannot use contract! from TensorAlgebra/src/contract/contract_matricize/contract.jl
# as it calls _mul!, which I should not overload.
# TBD define fallback _mul!(::AbstractArray, ::AbstractArray, ::AbstractArray) in TensorAlgebra?
function TensorAlgebra.contract!(
  ::Matricize,
  a_dest::FusionTensor,
  ::BlockedPermutation{2},
  a1::FusionTensor,
  biperm1::BlockedPermutation{2},
  a2::FusionTensor,
  biperm2::BlockedPermutation{2},
  α::Number,
  β::Number,
)
  a1_perm = permutedims(a1, biperm1)
  a2_perm = permutedims(a2, biperm2)
  mul!(a_dest, a1_perm, a2_perm, α, β)
  return a_dest
end
