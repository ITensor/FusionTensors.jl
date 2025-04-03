# This file defines unitaries to be used in permutedims

using BlockArrays: Block, findblock
using LRUCache: LRU

using GradedArrays.SymmetrySectors: quantum_dimension

const unitary_cache = LRU{Any,Float64}(; maxsize=10000)  # TBD size

# ======================================  Interface  =======================================
function compute_unitary(
  new_ft::FusionTensor{T,N}, old_ft::FusionTensor{T,N}, flatperm::NTuple{N,Int}
) where {T,N}
  return compute_unitary_clebsch_gordan(new_ft, old_ft, flatperm)
end

# ===========================  Constructor from Clebsch-Gordan  ============================
function overlap_fusion_trees(
  old_trees::Tuple{SectorFusionTree{S},SectorFusionTree{S}},
  new_trees::Tuple{SectorFusionTree{S},SectorFusionTree{S}},
  flatperm::Tuple{Vararg{Integer}},
) where {S}
  old_proj = contract_singlet_projector(old_trees...)
  new_proj = contract_singlet_projector(new_trees...)
  a = contract((), new_proj, flatperm, old_proj, ntuple(identity, ndims(new_proj)))
  return a[] / quantum_dimension(root_sector(first(new_trees)))
end

function cached_unitary_coeff(
  old_trees::Tuple{SectorFusionTree{S},SectorFusionTree{S}},
  new_trees::Tuple{SectorFusionTree{S},SectorFusionTree{S}},
  flatperm::Tuple{Vararg{Integer}},
) where {S}
  get!(unitary_cache, (old_trees..., new_trees..., flatperm)) do
    overlap_fusion_trees(old_trees, new_trees, flatperm)
  end
end

function compute_unitary_clebsch_gordan(
  new_ft::FusionTensor{T,N}, old_ft::FusionTensor{T,N}, flatperm::NTuple{N,Int}
) where {T,N}
  unitary = Dict{
    Tuple{keytype(trees_block_mapping(old_ft)),keytype(trees_block_mapping(new_ft))},Float64
  }()
  for old_trees in keys(trees_block_mapping(old_ft))
    old_outer = Tuple(findblock(old_ft, old_trees...))
    swapped_old_block = Block(getindex.(Ref(Tuple(old_outer)), flatperm))
    for new_trees in keys(trees_block_mapping(new_ft))
      swapped_old_block != findblock(new_ft, new_trees...) && continue
      unitary[old_trees, new_trees] = cached_unitary_coeff(old_trees, new_trees, flatperm)
    end
  end
  return unitary
end

# =================================  Constructor from 6j  ==================================
# dummy
