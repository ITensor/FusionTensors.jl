# This file defines fusion trees for any abelian or non-abelian group

# TBD
# compatibility with TensorKit conventions?

#
# A fusion tree tensor is a N+1 legs Array that corresponds to the projector defined by the
# associated fusion tree.
#
#      dim_sec  struct_mult_sec
#           \  /
#            \/
#            /
#           /
#          /\
#         /  \
#        /\   \
#       /  \   \
#     dim1 dim2 dim3
#
#
# It is convenient to "merge the tree leaves" by merging together the dimension legs to
# yield a 3-dim tensor with size (dim1*dim2*...*dimN, dim_sec, struct_mult_sec)
#
#             ---------------------------
#             |             |           |
#       dim1*dim2*dim3   dim_sec    struct_mult_sec
#
#
# convention: the trees are not normalized, i.e they do not define a projector on a given
# sector but carry a scaling factor sqrt(dim_sec)
#
# convention: irreps are already dualed if needed, arrows do not affect them. They only
# affect the basis on which the tree projects for self-dual irreps.
#
#
# The interface uses AbstractGradedUnitRanges as input for interface simplicity
# however only blocklabels are used and blocklengths are never read.

using SymmetrySectors:
  ⊗, AbelianStyle, AbstractSector, NotAbelianStyle, SectorProduct, SymmetryStyle, arguments

# ===================================  Utility tools  ======================================
function braid_tuples(t1::Tuple{Vararg{<:Any,N}}, t2::Tuple{Vararg{<:Any,N}}) where {N}
  t12 = (t1, t2)
  nested = ntuple(i -> getindex.(t12, i), N)
  return TensorAlgebra.flatten_tuples(nested)
end

# compute Kronecker product of fusion trees
# more efficient with recursive construction
trees_kron(a, b, c...) = trees_kron(trees_kron(a, b), c...)

function trees_kron(a, b)
  return map(
    ((t1, t2),) -> _tensor_kron(t1, t2), Iterators.flatten((Iterators.product(a, b),),)
  )
end

# LinearAlgebra.kron does not allow input for ndims>2
function _tensor_kron(a::AbstractArray{<:Any,N}, b::AbstractArray{<:Any,N}) where {N}
  t1 = ntuple(_ -> 1, N)
  sha = braid_tuples(size(a), t1)
  shb = braid_tuples(t1, size(b))
  c = reshape(a, sha) .* reshape(b, shb)
  return reshape(c, size(a) .* size(b))
end

# ================================  High level interface  ==================================
function merge_tree_leaves(a::AbstractArray)
  shape_3leg = (prod(size(a)[begin:(end - 2)]), size(a, ndims(a) - 1), size(a, ndims(a)))
  return reshape(a, shape_3leg)
end

function unmerge_tree_leaves(
  tree::AbstractArray{<:Real,3}, irreps::NTuple{<:Any,<:AbstractSector}
)
  irreps_shape = quantum_dimension.(irreps)
  return unmerge_tree_leaves(tree, irreps_shape)
end

function unmerge_tree_leaves(tree::AbstractArray{<:Real,3}, irreps_shape::NTuple{<:Any,Int})
  new_shape = (irreps_shape..., size(tree, 2), size(tree, 3))
  return reshape(tree, new_shape)
end

function get_fusion_tree_tensors!(
  cache::Dict{NTuple{N,Int},<:Vector{A}},
  it::NTuple{N,Int},
  legs::NTuple{N,AbstractGradedUnitRange},
  allowed_sectors::Vector{<:AbstractSector},
) where {N,A<:Array{<:Real}}
  get!(cache, it) do
    tree_arrows = isdual.(legs)
    irreps = getindex.(blocklabels.(legs), it)
    return compute_pruned_fusion_tree_tensors(A, irreps, tree_arrows, allowed_sectors)
  end
end

# explicitly cast trees to 3 leg format
function compute_pruned_fusion_tree_tensors(
  ::Type{<:Array{<:Real,3}},
  irreps::NTuple{N,<:AbstractSector},
  tree_arrows::NTuple{N,Bool},
  target_sectors::Vector{<:AbstractSector},
) where {N}
  return merge_tree_leaves.(
    compute_pruned_fusion_tree_tensors(Any, irreps, tree_arrows, target_sectors)
  )
end

function compute_pruned_fusion_tree_tensors(
  ::Type,
  irreps::NTuple{N,<:AbstractSector},
  tree_arrows::NTuple{N,Bool},
  target_sectors::Vector{<:AbstractSector},
) where {N}

  # it is possible to prune trees during the construction process and to avoid constructing
  # trees that will never fuse to target_sectors
  # currently this is not implemented and no pruning is done inside fusion_trees
  tree_irreps_pairs = fusion_tree_tensors(irreps, tree_arrows)
  tree_irreps = first.(tree_irreps_pairs)
  trees = last.(tree_irreps_pairs)

  # pruning is only done here by discarding irreps that are not in target_sectors
  # also insert dummy trees in sectors that did not appear in the fusion product of irreps
  irreps_dims = quantum_dimension.(irreps)
  trees_sector = [   # fill with dummy
    zeros((irreps_dims..., quantum_dimension(sec), 0)) for sec in target_sectors
  ]

  # set trees at their correct position
  for (i, s) in enumerate(target_sectors)
    j = findfirst(==(s), tree_irreps)
    if !isnothing(j)
      trees_sector[i] = trees[j]
    end
  end
  return trees_sector
end

# ================================  Low level interface  ===================================
function fusion_tree_tensors(::Tuple{}, ::Tuple{})
  return [TrivialSector() => ones((1, 1))]
end

function fusion_tree_tensors(
  irreps::NTuple{N,<:SectorProduct}, tree_arrows::NTuple{N,Bool}
) where {N}
  # construct fusion_tree(SectorProduct) as kron(fusion_trees(inner_sectors))

  argument_irreps = arguments.(irreps)
  n_args = length(first(argument_irreps))

  # construct fusion tree for each sector
  transposed_args = ntuple(s -> getindex.(argument_irreps, s), n_args)
  sector_trees_irreps = map(a -> fusion_tree_tensors(a, tree_arrows), transposed_args)

  # reconstruct sector for each product tree
  T = eltype(argument_irreps)
  fused_arguments = broadcast.(first, sector_trees_irreps)
  tree_irreps = map(
    SectorProduct ∘ T, Iterators.flatten((Iterators.product(fused_arguments...),))
  )

  # compute Kronecker product of fusion trees
  trees = trees_kron(broadcast.(last, sector_trees_irreps)...)

  # sort irreps. Each sector is sorted, permutation is obtained by reversing loop order
  perm = sortperm(tree_irreps)
  permute!(tree_irreps, perm)
  permute!(trees, perm)
  return tree_irreps .=> trees
end

function fusion_tree_tensors(
  irreps::NTuple{N,<:AbstractSector}, tree_arrows::NTuple{N,Bool}
) where {N}
  return fusion_tree_tensors(SymmetryStyle(first(irreps)), irreps, tree_arrows)
end

# =====================================  Internals  ========================================

# fusion tree for an Abelian group is trivial
# it does not depend on arrow directions
function fusion_tree_tensors(::AbelianStyle, irreps::Tuple, ::Tuple)
  irrep_prod = reduce(⊗, irreps)
  return [irrep_prod => ones(ntuple(_ -> 1, length(irreps) + 2))]
end

function build_children_trees(
  parent_tree::Matrix,
  parent_irrep::AbstractSector,
  level_irrep::AbstractSector,
  level_arrow::Bool,
  inner_multiplicity::Integer,
  sec::AbstractSector,
)
  sector_trees = typeof(parent_tree)[]
  for inner_mult_index in 1:inner_multiplicity
    cgt_inner_mult = clebsch_gordan_tensor(
      parent_irrep, level_irrep, sec, false, level_arrow, inner_mult_index
    )
    dim_parent_irrep, dim_level_irrep, dim_sec = size(cgt_inner_mult)
    tree =
      parent_tree * reshape(cgt_inner_mult, (dim_parent_irrep, dim_level_irrep * dim_sec))
    child_tree = reshape(tree, (size(parent_tree, 1) * dim_level_irrep, dim_sec))
    push!(sector_trees, child_tree)
  end
  return sector_trees
end

function build_children_trees(
  parent_tree::Matrix,
  parent_irrep::AbstractSector,
  level_irrep::AbstractSector,
  level_arrow::Bool,
)
  children_trees = typeof(parent_tree)[]
  children_irreps = typeof(parent_irrep)[]
  rep = fusion_product(parent_irrep, level_irrep)
  for (inner_multiplicity, sec) in zip(blocklengths(rep), blocklabels(rep))
    sector_trees = build_children_trees(
      parent_tree, parent_irrep, level_irrep, level_arrow, inner_multiplicity, sec
    )
    append!(children_trees, sector_trees)
    append!(children_irreps, repeat([sec], inner_multiplicity))
  end
  return children_trees, children_irreps
end

function build_next_level_trees(
  parent_trees::Vector,
  parent_trees_irreps::Vector,
  level_irrep::AbstractSector,
  level_arrow::Bool,
)
  next_level_trees = empty(parent_trees)
  next_level_irreps = empty(parent_trees_irreps)
  for (parent_tree, parent_irrep) in zip(parent_trees, parent_trees_irreps)
    children_trees, children_irreps = build_children_trees(
      parent_tree, parent_irrep, level_irrep, level_arrow
    )
    append!(next_level_trees, children_trees)
    append!(next_level_irreps, children_irreps)
  end
  return next_level_trees, next_level_irreps
end

function build_trees(trees::Vector, tree_irreps::Vector, irreps::Tuple, tree_arrows::Tuple)
  next_level_trees, next_level_irreps = build_next_level_trees(
    trees, tree_irreps, first(irreps), first(tree_arrows)
  )
  return build_trees(next_level_trees, next_level_irreps, irreps[2:end], tree_arrows[2:end])
end

function build_trees(trees::Vector, tree_irreps::Vector, ::Tuple{}, ::Tuple{})
  return trees, tree_irreps
end

function compute_thin_trees(irreps::Tuple, tree_arrows::Tuple)
  # init from trivial, NOT from first(irreps) to get first arrow correct
  thin_trees = [ones((1, 1))]
  tree_irreps = [trivial(first(irreps))]
  return build_trees(thin_trees, tree_irreps, irreps, tree_arrows)
end

function cat_thin_trees(
  thin_trees::Vector, uncat_tree_irreps::Vector, sector_irrep::AbstractSector
)
  indices_irrep = findall(==(sector_irrep), uncat_tree_irreps)
  thin_trees_irrep = getindex.(Ref(thin_trees), indices_irrep)
  thick_shape = (size(first(thin_trees_irrep))..., length(indices_irrep))
  return reshape(reduce(hcat, thin_trees_irrep), thick_shape)
end

function cat_thin_trees(thin_trees::Vector, uncat_tree_irreps::Vector)
  # cat trees fusing on the same irrep
  tree_irreps = sort(unique(uncat_tree_irreps))
  thick_trees = map(
    irrep -> cat_thin_trees(thin_trees, uncat_tree_irreps, irrep), tree_irreps
  )
  return thick_trees, tree_irreps
end

# arrow direction is needed to define non-trivial CG tensor
function fusion_tree_tensors(::NotAbelianStyle, irreps::Tuple, tree_arrows::Tuple)
  # compute "thin" trees: 1 tree = fuses on ONE irrep
  thin_trees, uncat_tree_irreps = compute_thin_trees(irreps, tree_arrows)

  # cat thin trees into "thick" trees
  thick_mergedleaves_trees, tree_irreps = cat_thin_trees(thin_trees, uncat_tree_irreps)

  irrep_dims = quantum_dimension.(irreps)
  thick_trees = map(tree -> unmerge_tree_leaves(tree, irrep_dims), thick_mergedleaves_trees)
  return tree_irreps .=> thick_trees
end

Base.convert(T::Type{<:Array}, f::FusionTree) = convert(T, to_tensor(f))

to_tensor(::FusionTree{<:Any,0}) = ones(1)

function to_tensor(f::FusionTree)
  # init with dummy trivial leg to get arrow correct and deal with size-1 case
  cgt1 = clebsch_gordan_tensor(
    trivial(eltype(f)),
    first(leaves(f)),
    first(branch_sectors(f)),
    false,
    first(arrows(f)),
    1,
  )
  return grow_tensor_tree(cgt1[1, :, :], f)
end

function contract_clebsch_gordan(
  tree_tensor::AbstractArray{<:Real,N}, cgt::AbstractArray{<:Real,3}
) where {N}
  return contract(
    (ntuple(identity, N - 1)..., N + 1, N + 2),
    tree_tensor,
    ntuple(identity, N),
    cgt,
    (N, N + 1, N + 2),
  )
end

function grow_tensor_tree(tree_tensor::AbstractArray{<:Real,1}, ::FusionTree{<:Any,1})
  return tree_tensor
end

function grow_tensor_tree(tree_tensor::AbstractArray{<:Real,N}, f::FusionTree) where {N}
  cgt = clebsch_gordan_tensor(
    branch_sectors(f)[N - 1],
    leaves(f)[N],
    branch_sectors(f)[N],
    false,
    arrows(f)[N],
    outer_multiplicty_indices(f)[N - 1],
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
    last(outer_multiplicty_indices(f)),
  )
  return contract_clebsch_gordan(tree_tensor, cgt)
end
