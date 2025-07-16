# This file defines permutedims for a FusionTensor

using BlockArrays: blocklengths
using Strided: Strided, @strided

using GradedArrays: AbelianStyle, NotAbelianStyle, SymmetryStyle, checkspaces
using TensorAlgebra: AbstractBlockPermutation, permmortar

# permutedims with 1 tuple of 2 separate tuples
function fusiontensor_permutedims(ft, new_leg_dims::Tuple{Tuple,Tuple})
  return fusiontensor_permutedims(ft, new_leg_dims...)
end

function fusiontensor_permutedims!(ftdst, ftsrc, new_leg_dims::Tuple{Tuple,Tuple})
  return fusiontensor_permutedims!(ftdst, ftsrc, new_leg_dims...)
end

# permutedims with 2 separate tuples
function fusiontensor_permutedims(ft, new_codomain_dims::Tuple, new_domain_dims::Tuple)
  biperm = permmortar((new_codomain_dims, new_domain_dims))
  return fusiontensor_permutedims(ft, biperm)
end

function fusiontensor_permutedims!(
  ftdst, ftsrc, new_codomain_dims::Tuple, new_domain_dims::Tuple
)
  biperm = permmortar((new_codomain_dims, new_domain_dims))
  return fusiontensor_permutedims!(ftdst, ftsrc, biperm)
end

# permutedims with BlockedPermutation
function fusiontensor_permutedims(ft, biperm::AbstractBlockPermutation{2})
  ndims(ft) == length(biperm) || throw(ArgumentError("Invalid permutation length"))
  ftdst = similar(ft, axes(ft)[biperm])
  fusiontensor_permutedims!(ftdst, ft, biperm)
  return ftdst
end

function fusiontensor_permutedims!(ftdst, ftsrc, biperm::AbstractBlockPermutation{2})
  ndims(ftsrc) == length(biperm) || throw(ArgumentError("Invalid permutation length"))
  blocklengths(axes(ftdst)) == blocklengths(biperm) ||
    throw(ArgumentError("Destination tensor does not match bipermutation"))
  checkspaces(axes(ftdst), axes(ftsrc)[biperm])

  # early return for identity operation. Also handle tricky 0-dim case.
  if ndims_codomain(ftdst) == ndims_codomain(ftsrc)  # compile time
    if Tuple(biperm) == ntuple(identity, ndims(ftdst))
      copy!(data_matrix(ftdst), data_matrix(ftsrc))
      return ftdst
    end
  end
  return _fusiontensor_permutedims!(SymmetryStyle(ftdst), ftdst, ftsrc, Tuple(biperm))
end

# ===============================   Internal   =============================================
function _fusiontensor_permutedims!(::AbelianStyle, ftdst, ftsrc, flatperm)
  # abelian case: all unitary blocks are 1x1 identity matrices
  # compute_unitary is only called to get block positions
  unitary = compute_unitary(ftdst, ftsrc, flatperm)
  for ((old_trees, new_trees), _) in unitary
    new_block = view(ftdst, new_trees...)
    old_block = view(ftsrc, old_trees...)
    @strided new_block .= permutedims(old_block, flatperm)
  end
  return ftdst
end

function _fusiontensor_permutedims!(::NotAbelianStyle, ftdst, ftsrc, flatperm)
  foreach(m -> fill!(m, zero(eltype(ftdst))), eachstoredblock(data_matrix(ftdst)))
  unitary = compute_unitary(ftdst, ftsrc, flatperm)
  for ((old_trees, new_trees), coeff) in unitary
    new_block = view(ftdst, new_trees...)
    old_block = view(ftsrc, old_trees...)
    @strided new_block .+= coeff .* permutedims(old_block, flatperm)
  end
  return ftdst
end
