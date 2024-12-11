# This file defines permutedims for a FusionTensor

# permutedims with 1 tuple of 2 separate tuples
function fusiontensor_permutedims(ft::FusionTensor, new_leg_indices::Tuple{Tuple,Tuple})
  return fusiontensor_permutedims(ft, new_leg_indices...)
end

# permutedims with 2 separate tuples
function fusiontensor_permutedims(
  ft::FusionTensor, new_codomain_indices::Tuple, new_domain_indices::Tuple
)
  biperm = blockedperm(new_codomain_indices, new_domain_indices)
  return fusiontensor_permutedims(ft, biperm)
end

function fusiontensor_permutedims(ft::FusionTensor, biperm::BlockedPermutation{2})
  @assert ndims(ft) == length(biperm)

  # early return for identity operation. Do not copy. Also handle tricky 0-dim case.
  if ndims_codomain(ft) == first(blocklengths(biperm))  # compile time
    if Tuple(biperm) == ntuple(identity, ndims(ft))
      return ft
    end
  end

  new_codomain_legs, new_domain_legs = blockpermute(axes(ft), biperm)
  permuted = FusionTensor(eltype(ft), new_codomain_legs, new_domain_legs)
  permute_fusiontensor_data!(permuted, ft, Tuple(biperm))
  return permuted
end

function naive_permutedims(ft::FusionTensor, biperm::BlockedPermutation{2})
  @assert ndims(ft) == length(biperm)
  new_codomain_legs, new_domain_legs = blockpermute(axes(ft), biperm)

  # naive permute: cast to dense, permutedims, cast to FusionTensor
  arr = Array(ft)
  permuted_arr = permutedims(arr, Tuple(biperm))
  permuted = FusionTensor(permuted_arr, new_codomain_legs, new_domain_legs)
  return permuted
end

function permute_fusiontensor_data!(
  new_ft::FusionTensor{T,N}, old_ft::FusionTensor{T,N}, flatperm::NTuple{N,Integer}
) where {T,N}
  unitary = compute_unitary(new_ft, old_ft, flatperm)
  for p in unitary
    old_trees, new_trees = first(p)
    new_ft[new_trees] += last(p) * permutedims(old_ft[old_trees], flatperm)
  end
end
