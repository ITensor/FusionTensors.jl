# This file defines fusion trees for any abelian or non-abelian group

# TBD
# compatibility with TensorKit conventions?

#
# A fusion tree fuses N sectors sec1, secN  onto one sector fused_sec. A given set of
# sectors and arrow directions (as defined by a given outer block) contains several fusion
# trees that typically fuse to several sectors (in the abelian group case, there is only one)
# irrep in the fusion ring and each of them corresponds to a single "thin" fusion tree with
#
#
#
#             /
#          sec123
#           /\
#          /  \
#       sec12  \
#        /\     \
#       /  \     \
#     sec1 sec2  sec3
#
#
#
#
# convention: irreps are already dualed if needed, arrows do not affect them. They only
# affect the basis on which the tree projects for self-dual irreps.
#
#
# The interface uses AbstractGradedUnitRanges as input for interface simplicity
# however only blocklabels are used and blocklengths are never read.

using SymmetrySectors: SymmetrySectors

struct FusionTree{S,N,M}
  leaves::NTuple{N,S}  # TBD rename outer_sectors or leave_sectors?
  arrows::NTuple{N,Bool}
  root_sector::S
  branch_sectors::NTuple{M,S}  # M = N-1
  outer_multiplicty_indices::NTuple{M,Int}  # M = N-1

  # TBD could have branch_sectors with length N-2
  # currently first(branch_sectors) == first(leaves)
  # redundant but allows for simpler, generic grow_tree code

  function FusionTree(
    leaves, arrows, root_sector, branch_sectors, outer_multiplicty_indices
  )
    N = length(leaves)
    @assert length(branch_sectors) == max(0, N - 1)
    @assert length(outer_multiplicty_indices) == max(0, N - 1)
    return new{typeof(root_sector),length(leaves),length(branch_sectors)}(
      leaves, arrows, root_sector, branch_sectors, outer_multiplicty_indices
    )
  end
end

# getters
arrows(f::FusionTree) = f.arrows
leaves(f::FusionTree) = f.leaves
root_sector(f::FusionTree) = f.root_sector
branch_sectors(f::FusionTree) = f.branch_sectors
outer_multiplicty_indices(f::FusionTree) = f.outer_multiplicty_indices

# interface
Base.length(::FusionTree{<:Any,N}) where {N} = N
Base.isless(f1::FusionTree, f2::FusionTree) = isless(to_tuple(f1), to_tuple(f2))

function to_tuple(f::FusionTree)   # TBD defined as Base.Tuple(::FusionTree)?
  return (
    leaves(f)...,
    arrows(f)...,
    root_sector(f),
    branch_sectors(f)...,
    outer_multiplicty_indices(f)...,
  )
end

GradedUnitRanges.sector_type(::FusionTree{S}) where {S} = S  # TBD use different function name?
Base.eltype(::FusionTree{S}) where {S} = S

function build_trees(legs::Vararg{AbstractGradedUnitRange{LA}}) where {LA}
  # TBD when to impose LA to be the same for every leg?
  tree_arrows = isdual.(legs)
  sectors = blocklabels.(legs)
  return mapreduce(vcat, CartesianIndices(blocklength.(legs))) do it
    block_sectors = getindex.(sectors, Tuple(it))  # why not type stable?
    return build_trees(block_sectors, tree_arrows)
  end
end

function SymmetrySectors.:×(f1::FusionTree, f2::FusionTree)
  @assert arrows(f1) == arrows(f2)
  product_leaves = .×(leaves(f1), leaves(f2))
  product_root_sector = root_sector(f1) × root_sector(f2)
  product_branch_sectors = .×(branch_sectors(f1), branch_sectors(f2))
  product_outer_multiplicty_indices =
    outer_multiplicity_kron.(
      Base.tail(leaves(f1)),
      branch_sectors(f1),
      (Base.tail(branch_sectors(f1))..., root_sector(f1)),
      outer_multiplicty_indices(f1),
      outer_multiplicty_indices(f2),
    )
  return FusionTree(
    product_leaves,
    arrows(f1),
    product_root_sector,
    product_branch_sectors,
    product_outer_multiplicty_indices,
  )
end

function outer_multiplicity_kron(
  sec1, sec2, fused, outer_multiplicity1, outer_multiplicity2
)
  full_space = fusion_product(sec1, sec2)
  nsymbol = blocklengths(full_space)[findfirst(==(fused), blocklabels(full_space))]
  linear_inds = LinearIndices((nsymbol, outer_multiplicity2))
  return linear_inds[outer_multiplicity1, outer_multiplicity2]
end

function outer_multiplicity_split(sec1, sec2, fused, outer_multiplicity)
  args1 = SymmetrySectors.arguments(sec1)
  args2 = SymmetrySectors.arguments(sec2)
  args12 = SymmetrySectors.arguments(fused)
  nsymbols = map(zip(args1, args2, args12)) do (sec1, sec2, sec12)
    full_space = fusion_product(sec1, sec2)
    return blocklengths(full_space)[findfirst(==(sec12), blocklabels(full_space))]
  end
  return CartesianIndices(nsymbols)[outer_multiplicity]
end

# zero leg: need S to get sector type information
function FusionTree{S}() where {S<:AbstractSector}
  return FusionTree((), (), trivial(S), (), ())
end
function FusionTree{S}(::Tuple{}, ::Tuple{}) where {S<:AbstractSector}
  return FusionTree((), (), trivial(S), (), ())
end

# one leg
FusionTree(sect::AbstractSector, arrow::Bool) = FusionTree((sect,), (arrow,), sect, (), ())

# =====================================  Internals  ========================================
function grow_tree(
  parent_tree::FusionTree,
  branch_sector::AbstractSector,
  level_arrow::Bool,
  child_root_sector,
  outer_mult,
)
  child_leaves = (leaves(parent_tree)..., branch_sector)
  child_arrows = (arrows(parent_tree)..., level_arrow)
  child_branch_sectors = (branch_sectors(parent_tree)..., root_sector(parent_tree))
  child_outer_mul = (outer_multiplicty_indices(parent_tree)..., outer_mult)
  return FusionTree(
    child_leaves, child_arrows, child_root_sector, child_branch_sectors, child_outer_mul
  )
end

function grow_tree(
  parent_tree::FusionTree, branch_sector::AbstractSector, level_arrow::Bool
)
  new_space = fusion_product(root_sector(parent_tree), branch_sector)
  return mapreduce(vcat, zip(blocklabels(new_space), blocklengths(new_space))) do (la, n)
    return [
      grow_tree(parent_tree, branch_sector, level_arrow, la, outer_mult) for
      outer_mult in 1:n
    ]
  end
end

function build_trees(old_trees::Vector, sectors_to_fuse::Tuple, arrows_to_fuse::Tuple)
  next_level_trees = mapreduce(vcat, old_trees) do tree
    return grow_tree(tree, first(sectors_to_fuse), first(arrows_to_fuse))
  end
  return build_trees(
    next_level_trees, Base.tail(sectors_to_fuse), Base.tail(arrows_to_fuse)
  )
end

function build_trees(trees::Vector, ::Tuple{}, ::Tuple{})
  return trees
end

function build_trees(
  sectors_to_fuse::NTuple{N,S}, arrows_to_fuse::NTuple{N,Bool}
) where {N,S<:AbstractSector}
  trees = [FusionTree(first(sectors_to_fuse), first(arrows_to_fuse))]
  return build_trees(trees, Base.tail(sectors_to_fuse), Base.tail(arrows_to_fuse))
end
