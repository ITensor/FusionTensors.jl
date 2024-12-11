# This file defines interface to cast from and to generic array

using BlockArrays: AbstractBlockArray, BlockedArray, blockedrange, blocklengths, findblock

using BlockSparseArrays: BlockSparseArrays, BlockSparseArray
using GradedUnitRanges: AbstractGradedUnitRange, blocklabels
using SymmetrySectors: block_dimensions, quantum_dimension
using TensorAlgebra: contract

# =================================  High level interface  =================================

#### cast from array to symmetric
function FusionTensor(
  array::AbstractArray,
  codomain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
  domain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
)
  return cast_from_array(array, codomain_legs, domain_legs)
end

#### cast from symmetric to array
function BlockSparseArrays.BlockSparseArray(ft::FusionTensor)
  return cast_to_array(ft)
end

# =================================  Low level interface  ==================================
function cast_from_array(
  array::AbstractArray,
  codomain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
  domain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
)
  bounds = block_dimensions.((codomain_legs..., domain_legs...))
  blockarray = BlockedArray(array, bounds...)
  return cast_from_array(blockarray, codomain_legs, domain_legs)
end

function cast_from_array(
  blockarray::AbstractBlockArray,
  codomain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
  domain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
)
  # input validation
  if length(codomain_legs) + length(domain_legs) != ndims(blockarray)  # compile time
    throw(codomainError("legs are incompatible with array ndims"))
  end
  if quantum_dimension.((codomain_legs..., domain_legs...)) != size(blockarray)
    throw(codomainError("legs dimensions are incompatible with array"))
  end

  ft = FusionTensor(eltype(blockarray), codomain_legs, domain_legs)
  for (f1, f2) in keys(trees_block_mapping(ft))
    b = findblock(ft, f1, f2)
    ft[f1, f2] = contract_fusion_trees(blockarray[b], f1, f2)
  end
  return ft
end

function cast_to_array(ft::FusionTensor)
  bounds = block_dimensions.((codomain_axes(ft)..., domain_axes(ft)...))
  bsa = BlockSparseArray{eltype(ft)}(blockedrange.(bounds))
  for (f1, f2) in keys(trees_block_mapping(ft))
    b = findblock(ft, f1, f2)
    bsa[b] = bsa[b] + contract_fusion_trees(ft, f1, f2)  # init block when needed
  end
  return bsa
end

# =====================================  Internals  ========================================

#-----------------------------------   misc tools  ----------------------------------------

function degen_dims_shape(dense_shape::Tuple{Vararg{Int}}, f::FusionTree)
  dims = quantum_dimension.(leaves(f))
  mults = dense_shape .÷ dims
  return braid_tuples(mults, dims)
end

function split_degen_dims(array_block::AbstractArray, f1::FusionTree, f2::FusionTree)
  array_block_split_shape = (
    degen_dims_shape(size(array_block)[begin:length(f1)], f1)...,
    degen_dims_shape(size(array_block)[(length(f1) + 1):end], f2)...,
  )
  return reshape(array_block, array_block_split_shape)
end

function merge_degen_dims(split_array_block::AbstractArray)
  s0 = size(split_array_block)
  array_shape =
    ntuple(i -> s0[2 * i - 1], length(s0) ÷ 2) .* ntuple(i -> s0[2 * i], length(s0) ÷ 2)
  array_block = reshape(split_array_block, array_shape)
  return array_block
end

#----------------------------------  cast from array ---------------------------------------

function contract_fusion_trees(array_block::AbstractArray, f1::FusionTree, f2::FusionTree)
  # start from an array outer block with e.g. N=6 axes divided into N_DO=3 ndims_codomain
  # and N_CO=3 ndims_domain. Each leg k can be decomposed as a product of external an
  # multiplicity extk and a quantum dimension dimk
  #
  #        ------------------------------array_block-------------------------------
  #        |             |             |              |               |           |
  #       ext1*dim1   ext2*dim2     ext3*dim3      ext4*dim4       ext5*dim5   ext6*dim6
  #

  # each leg of this this array outer block can now be opened to form a 2N-dims tensor.
  # note that this 2N-dims form is only defined at the level of the outer block,
  # not for a larger block.
  #
  #        ------------------------------split_array_block-------------------------
  #        |             |              |             |             |             |
  #       / \           / \            / \           / \           / \           / \
  #      /   \         /   \          /   \         /   \         /   \         /   \
  #    ext1  dim1    ext2  dim2     ext3  dim3    ext4  dim4    ext5  dim5    ext6 dim6
  #
  N = ndims(array_block)

  split_array_block = split_degen_dims(array_block, f1, f2)
  dim_sec = quantum_dimension(root_sector(f1))
  p = contract_singlet_projector(f1, f2)

  #        ------------------------------split_array_block-------------------------
  #        |             |              |             |             |             |
  #       / \           / \            / \           / \           / \           / \
  #      /   \         /   \          /   \         /   \         /   \         /   \
  #    ext1  dim1    ext2  dim2     ext3  dim3    ext4  dim4    ext5  dim5    ext6 dim6
  #              \           |            |                 \__         |            |
  #               \________  |            |                       \__   |            |
  #                         \|            |                          \ _|            |
  #                           \___________|                              \___________|
  #                                        \                                         |
  #                                         \----------------dim_sec---------------- /
  return contract(
    ntuple(i -> 2 * i - 1, N),
    split_array_block,
    ntuple(identity, 2 * N),
    p,
    ntuple(i -> 2 * i, N),
    1 / dim_sec,  # normalization factor
  )
end

function contract_singlet_projector(f1::FusionTree, f2::FusionTree)
  f1_array = convert(Array, f1)
  f2_array = convert(Array, f2)
  N_CO = length(f1)
  N_DO = length(f2)
  return contract(
    ntuple(identity, N_CO + N_DO),
    f1_array,
    (ntuple(identity, N_CO)..., N_CO + N_DO + 1),
    f2_array,
    (ntuple(i -> i + N_CO, N_DO)..., N_CO + N_DO + 1),
  )
end

#-----------------------------------  cast to array ----------------------------------------
function contract_fusion_trees(ft::FusionTensor, f1::FusionTree, f2::FusionTree)
  N = ndims(ft)
  charge_block = reshape(view(ft, f1, f2), :, 1)
  p = contract_singlet_projector(f1, f2)

  # TODO use contract once it supports outer product
  swapped = charge_block * reshape(p, 1, :)
  b = findblock(ft, f1, f2)
  block_shape = (
    ntuple(i -> blocklengths(axes(ft, i))[Int(Tuple(b)[i])], N)...,
    ntuple(i -> quantum_dimension(blocklabels(axes(ft, i))[Int(Tuple(b)[i])]), N)...,
  )
  perm = braid_tuples(ntuple(identity, N), ntuple(i -> i + N, N))
  split_array_block = permutedims(reshape(swapped, block_shape), perm)

  return merge_degen_dims(split_array_block)
end
