# This file defines permutedims for a FusionTensor

using BlockArrays: blocklengths
using Strided: Strided, @strided

using TensorAlgebra: BlockedPermutation, blockedperm, blockpermute

function naive_permutedims(ft, biperm::BlockedPermutation{2})
  @assert ndims(ft) == length(biperm)
  new_codomain_legs, new_domain_legs = blockpermute(axes(ft), biperm)

  # naive permute: cast to dense, permutedims, cast to FusionTensor
  arr = Array(ft)
  permuted_arr = permutedims(arr, Tuple(biperm))
  permuted = FusionTensor(permuted_arr, new_codomain_legs, new_domain_legs)
  return permuted
end

# permutedims with 1 tuple of 2 separate tuples
function fusiontensor_permutedims(ft, new_leg_indices::Tuple{Tuple,Tuple})
  return fusiontensor_permutedims(ft, new_leg_indices...)
end

# permutedims with 2 separate tuples
function fusiontensor_permutedims(
  ft, new_codomain_indices::Tuple, new_domain_indices::Tuple
)
  biperm = blockedperm(new_codomain_indices, new_domain_indices)
  return fusiontensor_permutedims(ft, biperm)
end

function fusiontensor_permutedims(ft, biperm::BlockedPermutation{2})
  @assert ndims(ft) == length(biperm)

  # early return for identity operation. Do not copy. Also handle tricky 0-dim case.
  if ndims_codomain(ft) == first(blocklengths(biperm))  # compile time
    if Tuple(biperm) == ntuple(identity, ndims(ft))
      return ft
    end
  end

  new_codomain_legs, new_domain_legs = blockpermute(axes(ft), biperm)
  new_ft = FusionTensor(eltype(ft), new_codomain_legs, new_domain_legs)
  fusiontensor_permutedims!(new_ft, ft, Tuple(biperm))
  return new_ft
end

function fusiontensor_permutedims!(
  new_ft::FusionTensor{T,N}, old_ft::FusionTensor{T,N}, flatperm::NTuple{N,Integer}
) where {T,N}
  unitary = compute_unitary(new_ft, old_ft, flatperm)
  for p in unitary
    old_trees, new_trees = first(p)
    new_block = view(new_ft, new_trees)
    old_block = view(old_ft, old_trees)
    @strided new_block .+= last(p) .* permutedims(old_block, flatperm)
  end
end