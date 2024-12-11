# This file defines unitaries to be used in permutedims

using BlockArrays: Block, findblock
using LRUCache: LRU

using SymmetrySectors: quantum_dimension

const unitary_cache = LRU{Any,AbstractMatrix}(; maxsize=10000)

# ======================================  Interface  =======================================
function compute_unitary(
  new_ft::FusionTensor{T,N}, old_ft::FusionTensor{T,N}, flatperm::NTuple{N,Int}
) where {T,N}
  return compute_unitary_clebsch_gordan(new_ft, old_ft, flatperm)
end

function unitary_key(codomain_legs, domain_legs, old_outer_block, biperm)
  legs = (codomain_legs..., domain_legs...)
  old_arrows = isdual.(legs)
  old_sectors = ntuple(i -> blocklabels(legs[i])[old_outer_block[i]], length(legs))
  return (old_arrows, old_sectors, length(codomain_legs), biperm)
end

# ===========================  Constructor from Clebsch-Gordan  ============================
function overlap_fusion_trees(
  old_trees::Tuple{FusionTree,FusionTree},
  new_trees::Tuple{FusionTree,FusionTree},
  flatperm::Tuple{Vararg{Integer}},
)
  old_proj = contract_singlet_projector(old_trees...)
  new_proj = contract_singlet_projector(new_trees...)
  a = contract((), new_proj, flatperm, old_proj, ntuple(identity, ndims(new_proj)))
  return a[] / quantum_dimension(root_sector(first(new_trees)))
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
      new_outer = findblock(new_ft, new_trees...)
      if swapped_old_block == new_outer
        unitary[old_trees, new_trees] = overlap_fusion_trees(old_trees, new_trees, flatperm)
      end
    end
  end
  return unitary
end

# =================================  Constructor from 6j  ==================================
# dummy
