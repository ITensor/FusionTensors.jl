# This file defines fusion trees for any abelian or non-abelian group

# TBD
# compatibility with TensorKit conventions?

using GradedUnitRanges:
  AbstractGradedUnitRange, GradedUnitRanges, fusion_product, isdual, sector_type
using SymmetrySectors: ×, AbstractSector, SectorProduct, SymmetrySectors, arguments, trivial
using TensorAlgebra: flatten_tuples

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

struct FusionTree{S,N,M}
  leaves::NTuple{N,S}  # TBD rename outer_sectors or leave_sectors?
  arrows::NTuple{N,Bool}
  root_sector::S
  branch_sectors::NTuple{M,S}  # M = N-1
  outer_multiplicity_indices::NTuple{M,Int}  # M = N-1

  # TBD could have branch_sectors with length N-2
  # currently first(branch_sectors) == first(leaves)
  # redundant but allows for simpler, generic grow_tree code

  function FusionTree(
    leaves, arrows, root_sector, branch_sectors, outer_multiplicity_indices
  )
    N = length(leaves)
    @assert length(branch_sectors) == max(0, N - 1)
    @assert length(outer_multiplicity_indices) == max(0, N - 1)
    return new{typeof(root_sector),length(leaves),length(branch_sectors)}(
      leaves, arrows, root_sector, branch_sectors, outer_multiplicity_indices
    )
  end
end

# getters
arrows(f::FusionTree) = f.arrows
leaves(f::FusionTree) = f.leaves
root_sector(f::FusionTree) = f.root_sector
branch_sectors(f::FusionTree) = f.branch_sectors
outer_multiplicity_indices(f::FusionTree) = f.outer_multiplicity_indices

# Base interface
Base.convert(T::Type{<:Array}, f::FusionTree) = convert(T, to_tensor(f))
Base.isless(f1::FusionTree, f2::FusionTree) = isless(to_tuple(f1), to_tuple(f2))
Base.length(::FusionTree{<:Any,N}) where {N} = N

# GradedUnitRanges interface
GradedUnitRanges.sector_type(::FusionTree{S}) where {S} = S

function build_trees(legs::Vararg{AbstractGradedUnitRange})
  tree_arrows = isdual.(legs)
  sectors = blocklabels.(legs)
  return mapreduce(vcat, CartesianIndices(blocklength.(legs))) do it
    block_sectors = getindex.(sectors, Tuple(it))  # why not type stable?
    return build_trees(block_sectors, tree_arrows)
  end
end

# SymmetrySectors interface
function SymmetrySectors.:×(f1::FusionTree, f2::FusionTree)
  @assert arrows(f1) == arrows(f2)
  product_leaves = .×(leaves(f1), leaves(f2))
  product_root_sector = root_sector(f1) × root_sector(f2)
  product_branch_sectors = .×(branch_sectors(f1), branch_sectors(f2))
  product_outer_multiplicity_indices =
    outer_multiplicity_kron.(
      Base.tail(leaves(f1)),
      branch_sectors(f1),
      (Base.tail(branch_sectors(f1))..., root_sector(f1)),
      outer_multiplicity_indices(f1),
      outer_multiplicity_indices(f2),
    )
  return FusionTree(
    product_leaves,
    arrows(f1),
    product_root_sector,
    product_branch_sectors,
    product_outer_multiplicity_indices,
  )
end

function SymmetrySectors.arguments(f::FusionTree{<:SectorProduct})
  transposed_indices =
    outer_multiplicity_split.(
      Base.tail(leaves(f)),
      branch_sectors(f),
      (Base.tail(branch_sectors(f))..., root_sector(f)),
      outer_multiplicity_indices(f),
    )
  arguments_root = arguments(root_sector(f))
  arguments_leaves = arguments.(leaves(f))
  arguments_branch_sectors = arguments.(branch_sectors(f))
  # TODO way to avoid explicit ntuple?
  # works fine for Tuple and NamedTuple SectorProduct
  return ntuple(
    i -> FusionTree(
      getindex.(arguments_leaves, i),
      arrows(f),
      arguments_root[i],
      getindex.(arguments_branch_sectors, i),
      getindex.(transposed_indices, i),
    ),
    length(arguments_root),
  )
end

function SymmetrySectors.arguments(f::FusionTree{<:SectorProduct,0})
  return map(arg -> FusionTree((), (), arg, (), ()), arguments(root_sector(f)))
end

function SymmetrySectors.arguments(f::FusionTree{<:SectorProduct,1})
  arguments_root = arguments(root_sector(f))
  arguments_leave = arguments(only(leaves(f)))
  # use map(keys) to stay agnostic with respect to SectorProduct implementation
  return map(
    k -> FusionTree((arguments_leave[k],), arrows(f), arguments_root[k], (), ()),
    keys(arguments_root),
  )
end

#
# =====================================  Internals  ========================================
#
# --------------- misc  ---------------
function to_tuple(f::FusionTree)
  return (
    leaves(f)...,
    arrows(f)...,
    root_sector(f),
    branch_sectors(f)...,
    outer_multiplicity_indices(f)...,
  )
end

# --------------- SectorProduct helper functions  ---------------
function outer_multiplicity_kron(
  sec1, sec2, fused, outer_multiplicity1, outer_multiplicity2
)
  n = nsymbol(sec1, sec2, fused)
  linear_inds = LinearIndices((n, outer_multiplicity2))
  return linear_inds[outer_multiplicity1, outer_multiplicity2]
end

# TODO move to GradedUnitRanges
function nsymbol(s1::AbstractSector, s2::AbstractSector, s3::AbstractSector)
  full_space = fusion_product(s1, s2)
  x = findfirst(==(s3), blocklabels(full_space))
  isnothing(x) && return 0  # OR labelled(0, s3)?
  return Int(blocklengths(full_space)[x])
end

function outer_multiplicity_split(
  sec1::S, sec2::S, fused::S, outer_mult_index::Integer
) where {S<:SectorProduct}
  args1 = arguments(sec1)
  args2 = arguments(sec2)
  args12 = arguments(fused)
  nsymbols = Tuple(map(nsymbol, args1, args2, args12))  # CartesianIndices requires explicit Tuple
  return Tuple(CartesianIndices(nsymbols)[outer_mult_index])
end

# --------------- Build trees  ---------------
# zero leg: need S to get sector type information
function FusionTree{S}() where {S<:AbstractSector}
  return FusionTree((), (), trivial(S), (), ())
end
function FusionTree{S}(::Tuple{}, ::Tuple{}) where {S<:AbstractSector}
  return FusionTree((), (), trivial(S), (), ())
end

# one leg
FusionTree(sect::AbstractSector, arrow::Bool) = FusionTree((sect,), (arrow,), sect, (), ())

function braid_tuples(t1::Tuple{Vararg{Any,N}}, t2::Tuple{Vararg{Any,N}}) where {N}
  t12 = (t1, t2)
  nested = ntuple(i -> getindex.(t12, i), N)
  return flatten_tuples(nested)
end

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
  child_outer_mul = (outer_multiplicity_indices(parent_tree)..., outer_mult)
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
  sectors_to_fuse::NTuple{N,<:AbstractSector}, arrows_to_fuse::NTuple{N,Bool}
) where {N}
  trees = [FusionTree(first(sectors_to_fuse), first(arrows_to_fuse))]
  return build_trees(trees, Base.tail(sectors_to_fuse), Base.tail(arrows_to_fuse))
end

# --------------- convert to Array  ---------------
to_tensor(::FusionTree{<:Any,0}) = ones(1)

function to_tensor(f::FusionTree)
  # init with dummy trivial leg to get arrow correct and deal with size-1 case
  cgt1 = clebsch_gordan_tensor(
    trivial(sector_type(f)), first(leaves(f)), first(leaves(f)), false, first(arrows(f)), 1
  )
  tree_tensor = cgt1[1, :, :]
  return grow_tensor_tree(tree_tensor, f)
end

#to_tensor(::FusionTree{<:SectorProduct,0}) = ones(1)
function to_tensor(f::FusionTree{<:SectorProduct})
  args = convert.(Array, arguments(f))
  return reduce(_tensor_kron, args)
end

# LinearAlgebra.kron does not allow input for ndims>2
function _tensor_kron(a::AbstractArray{<:Any,N}, b::AbstractArray{<:Any,N}) where {N}
  t1 = ntuple(_ -> 1, N)
  sha = braid_tuples(size(a), t1)
  shb = braid_tuples(t1, size(b))
  c = reshape(a, sha) .* reshape(b, shb)
  return reshape(c, size(a) .* size(b))
end

function contract_clebsch_gordan(tree_tensor::AbstractArray, cgt::AbstractArray)
  N = ndims(tree_tensor)
  return contract(
    (ntuple(identity, N - 1)..., N + 1, N + 2),
    tree_tensor,
    ntuple(identity, N),
    cgt,
    (N, N + 1, N + 2),
  )
end

# specialized code when branch_sector is empty
function grow_tensor_tree(tree_tensor::AbstractArray{<:Real,2}, ::FusionTree{<:Any,1})
  return tree_tensor
end

function grow_tensor_tree(tree_tensor::AbstractArray{<:Real,N}, f::FusionTree) where {N}
  cgt = clebsch_gordan_tensor(
    branch_sectors(f)[N - 1],
    leaves(f)[N],
    branch_sectors(f)[N],
    false,
    arrows(f)[N],
    outer_multiplicity_indices(f)[N - 1],
  )
  next_level_tree = contract_clebsch_gordan(tree_tensor, cgt)
  return grow_tensor_tree(next_level_tree, f)
end

function grow_tensor_tree(
  tree_tensor::AbstractArray{<:Real,N}, f::FusionTree{<:Any,N}
) where {N}
  cgt = clebsch_gordan_tensor(
    last(branch_sectors(f)),
    last(leaves(f)),
    root_sector(f),
    false,
    last(arrows(f)),
    last(outer_multiplicity_indices(f)),
  )
  return contract_clebsch_gordan(tree_tensor, cgt)
end
