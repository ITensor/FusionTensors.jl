module FusionTensors

using LinearAlgebra: LinearAlgebra, Adjoint, norm, tr

using BlockArrays:
  AbstractBlockArray,
  AbstractBlockMatrix,
  Block,
  BlockArray,
  BlockedArray,
  BlockMatrix,
  blockedrange,
  blocklength,
  blocklengths,
  blocks
using LRUCache: LRU

using BlockSparseArrays:
  AbstractBlockSparseMatrix,
  BlockSparseArrays,
  BlockSparseArray,
  BlockSparseMatrix,
  stored_indices,
  view!
using GradedUnitRanges:
  GradedUnitRanges,
  AbstractGradedUnitRange,
  blocklabels,
  blockmergesort,
  dual,
  fusion_product,
  gradedrange,
  isdual,
  labelled_blocks,
  sector_type,
  space_isequal,
  unlabel_blocks
using SymmetrySectors:
  AbstractSector, TrivialSector, block_dimensions, istrivial, quantum_dimension, trivial
using TensorAlgebra:
  TensorAlgebra,
  Algorithm,
  BlockedPermutation,
  blockedperm,
  blockpermute,
  contract,
  contract!

include("fusion_trees/fusiontree.jl")
include("fusion_trees/clebsch_gordan_tensors.jl")
include("fusion_trees/fusion_tree_tensors.jl")

include("fusiontensor/fusedaxes.jl")
include("fusiontensor/fusiontensor.jl")
include("fusiontensor/base_interface.jl")
include("fusiontensor/array_cast.jl")
include("fusiontensor/linear_algebra_interface.jl")
include("fusiontensor/tensor_algebra_interface.jl")
include("permutedims/unitaries.jl")
include("permutedims/permutedims.jl")
end
