# This file defines struct FusionTensor and constructors

using BlockArrays: AbstractBlockMatrix, BlockArrays, BlockIndexRange, blocklength, findblock

using BlockSparseArrays:
  AbstractBlockSparseMatrix, BlockSparseArray, eachblockstoredindex, to_block_indices
using GradedArrays:
  AbstractGradedUnitRange,
  blocklabels,
  blockmergesort,
  dual,
  flip,
  gradedrange,
  isdual,
  map_blocklabels,
  sector_type,
  space_isequal
using GradedArrays.SymmetrySectors: SectorProduct, TrivialSector
using TensorAlgebra: BlockedTuple, tuplemortar
using TensorProducts: tensor_product

struct FusionTensor{T,N,Axes,Mat,Mapping} <: AbstractArray{T,N}
  data_matrix::Mat
  axes::Axes
  trees_block_mapping::Mapping

  # inner constructor to impose constraints on types
  function FusionTensor(
    mat::AbstractMatrix,
    legs::BlockedTuple{2,<:Any,<:Tuple{Vararg{AbstractGradedUnitRange}}},
    trees_block_mapping::Dict,
  )
    S = sector_type(axes(mat, 1))
    @assert sector_type(axes(mat, 2)) === S
    @assert keytype(trees_block_mapping) <:
      Tuple{<:SectorFusionTree{S},<:SectorFusionTree{S}}
    @assert all(sector_type.(Tuple(legs)) .=== S)
    return new{
      eltype(mat),length(legs),typeof(legs),typeof(mat),typeof(trees_block_mapping)
    }(
      mat, legs, trees_block_mapping
    )
  end
end

# getters
data_matrix(ft::FusionTensor) = ft.data_matrix
trees_block_mapping(ft::FusionTensor) = ft.trees_block_mapping

# misc access
codomain_axes(ft::FusionTensor) = first(blocks(axes(ft)))
domain_axes(ft::FusionTensor) = last(blocks(axes(ft)))
ndims_codomain(ft::FusionTensor) = length(codomain_axes(ft))
ndims_domain(ft::FusionTensor) = length(domain_axes(ft))

matrix_size(ft::FusionTensor) = quantum_dimension.(axes(data_matrix(ft)))
matrix_row_axis(ft::FusionTensor) = first(axes(data_matrix(ft)))
matrix_column_axis(ft::FusionTensor) = last(axes(data_matrix(ft)))
function charge_block_size(ft::FusionTensor, f1::SectorFusionTree, f2::SectorFusionTree)
  b = Tuple(findblock(ft, f1, f2))
  return ntuple(i -> Int(length(axes(ft)[i][b[i]])), ndims(ft))
end

# GradedArrays interface
function GradedArrays.sector_type(
  ::Type{<:FusionTensor{<:Any,<:Any,<:Any,<:Any,<:Dict{<:Tuple{<:Any,F}}}}
) where {F}
  return sector_type(F)
end

# BlockArrays interface
function BlockArrays.findblock(ft::FusionTensor, f1::SectorFusionTree, f2::SectorFusionTree)
  # find outer block corresponding to fusion trees
  @assert typeof((f1, f2)) === keytype(trees_block_mapping(ft))
  b1 = find_sector_block.(leaves(f1), codomain_axes(ft))
  b2 = find_sector_block.(leaves(f2), domain_axes(ft))
  return Block(b1..., b2...)
end
# TBD move to GradedArrays? rename findfirst_sector?
function find_sector_block(s::AbstractSector, l::AbstractGradedUnitRange)
  return findfirst(==(s), blocklabels(l))
end

function sanitize_axes(raw_legs::Tuple{Vararg{AbstractGradedUnitRange}})
  legs = promote_sectors(typeof(first(raw_legs)), raw_legs)
  @assert all(check_unique_blocklabels.(legs))
  return legs
end
sanitize_axes(legs::BlockedTuple{2,(0, 0)}) = TrivialSector, legs
function sanitize_axes(raw_legs::BlockedTuple{2})
  flat_legs = sanitize_axes(Tuple(raw_legs))
  return sector_type(first(flat_legs)), BlockedTuple(flat_legs, Val(blocklengths(raw_legs)))
end

function check_unique_blocklabels(g::AbstractGradedUnitRange)
  return length(unique(blocklabels(g))) == blocklength(g)
end

function promote_sectors(
  ::Type{<:AbstractGradedUnitRange{LA}}, legs::Tuple{Vararg{AbstractGradedUnitRange{LA}}}
) where {LA}  # nothing to do
  return legs
end

function promote_sectors(
  ::Type{<:AbstractGradedUnitRange}, legs::Tuple{Vararg{AbstractGradedUnitRange}}
)
  T = promote_sector_type(legs)
  # fuse with trivial to insert all missing arguments inside each GradedAxis
  # avoid depending on SymmetrySectors internals
  s0 = trivial(T)
  return map_blocklabels.(s -> only(blocklabels(tensor_product(s0, s))), legs)
end

function promote_sector_type(legs)
  # fuse trivial sectors to produce unified type
  # avoid depending on SymmetrySectors internals
  return sector_type(tensor_product(trivial.(legs)...))
end

# initialize with already computed data_matrix
function FusionTensor(
  x,
  codomain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
  domain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
)
  return FusionTensor(x, tuplemortar((codomain_legs, domain_legs)))
end

function FusionTensor(mat::AbstractMatrix, legs::BlockedTuple{2})
  # init with empty data_matrix to construct trees_block_mapping
  ft = FusionTensor(eltype(mat), legs)
  @assert space_isequal(matrix_row_axis(ft), axes(mat, 1))
  @assert space_isequal(matrix_column_axis(ft), axes(mat, 2))
  for b in eachblockstoredindex(mat)
    @assert b in eachblockstoredindex(data_matrix(ft))  # check matrix block is allowed
    data_matrix(ft)[b] = mat[b]
  end
  return ft
end

function flip_domain(nonflipped_col_axis, nonflipped_trees_to_ranges)
  col_axis = dual(nonflipped_col_axis)
  domain_trees_to_ranges_mapping = Dict(
    map(((tree, v),) -> flip(tree) => v, collect(nonflipped_trees_to_ranges))
  )
  return col_axis, domain_trees_to_ranges_mapping
end

# empty matrix
function FusionTensor(elt::Type, raw_legs::BlockedTuple{2})
  S, legs = sanitize_axes(raw_legs)

  row_axis, codomain_trees_to_ranges = fuse_axes(S, first(blocks(legs)))
  col_axis, domain_trees_to_ranges = flip_domain(fuse_axes(S, dual.(last(blocks(legs))))...)

  mat = initialize_data_matrix(elt, row_axis, col_axis)
  tree_to_block_mapping = intersect_codomain_domain(
    codomain_trees_to_ranges, domain_trees_to_ranges
  )
  return FusionTensor(mat, legs, tree_to_block_mapping)
end

function fuse_axes(::Type{S}, ::Tuple{}) where {S<:AbstractSector}
  fused_axis = gradedrange([trivial(S) => 1])
  trees_to_ranges_mapping = Dict([SectorFusionTree{S}() => Block(1)[1:1]])
  return fused_axis, trees_to_ranges_mapping
end
function fuse_axes(::Type, outer_legs::Tuple{Vararg{AbstractGradedUnitRange}})
  fusion_trees_mult = fusion_trees_external_multiplicities(outer_legs)
  fused_leg, trees_to_ranges_mapping = compute_inner_ranges(fusion_trees_mult)
  return fused_leg, trees_to_ranges_mapping
end

function fusion_trees_external_multiplicities(
  outer_legs::Tuple{Vararg{AbstractGradedUnitRange}}
)
  return Iterators.flatten(
    block_fusion_trees_external_multiplicities.(Iterators.product(blocks.(outer_legs)...))
  )
end

function block_fusion_trees_external_multiplicities(it::Tuple{Vararg{AbstractUnitRange}})
  block_sectors = only.(blocklabels.(it))
  block_mult = prod(length.(it))
  return build_trees(block_sectors, isdual.(it)) .=> block_mult
end

function compute_inner_ranges(fusion_trees_mult)
  fused_leg = blockmergesort(
    gradedrange(root_sector.(first.(fusion_trees_mult)) .=> last.(fusion_trees_mult))
  )
  range_mapping = Dict{fieldtype(eltype(fusion_trees_mult), 1),typeof(Block(1)[1:1])}()
  fused_sectors = blocklabels(fused_leg)
  shifts = ones(Int, blocklength(fused_leg))
  for (f, m) in fusion_trees_mult
    s = root_sector(f)
    i = findfirst(==(s), fused_sectors)
    range_mapping[f] = Block(i)[shifts[i]:(shifts[i] + m - 1)]
    shifts[i] += m
  end
  return fused_leg, range_mapping
end

function to_blockindexrange(b1::BlockIndexRange{1}, b2::BlockIndexRange{1})
  t = (b1, b2)
  return Block(Block.(t))[to_block_indices.(t)...]
end

function intersect_codomain_domain(
  codomain_trees_to_ranges_mapping::Dict{<:SectorFusionTree,<:BlockIndexRange{1}},
  domain_trees_to_ranges_mapping::Dict{<:SectorFusionTree,<:BlockIndexRange{1}},
)
  return Dict(
    map(
      Iterators.filter(
        t -> root_sector(first(first(t))) == dual(root_sector(first(t[2]))),
        Iterators.product(codomain_trees_to_ranges_mapping, domain_trees_to_ranges_mapping),
      ),
    ) do t
      return first.(t) => to_blockindexrange(last.(t)...)
    end,
  )
end

function initialize_data_matrix(
  elt::Type{<:Number},
  mat_row_axis::AbstractGradedUnitRange,
  mat_col_axis::AbstractGradedUnitRange,
)
  # non-abelian fusion trees have float eltype: need compatible type
  promoted = promote_type(elt, fusiontree_eltype(sector_type(mat_row_axis)))
  mat = BlockSparseArray{promoted}(undef, mat_row_axis, mat_col_axis)
  initialize_allowed_sectors!(mat)
  return mat
end

function initialize_allowed_sectors!(mat::AbstractMatrix)
  row_sectors = blocklabels(axes(mat, 1))
  col_sectors = blocklabels(dual(axes(mat, 2)))
  row_block_indices = findall(in(col_sectors), row_sectors)
  col_block_indices = findall(in(row_sectors), col_sectors)
  for rc in zip(row_block_indices, col_block_indices)
    mat[Block(rc)] = mat[Block(rc)]
  end
end

checkaxes_dual(axes1, axes2) = checkaxes(axes1, dual.(axes2))
function checkaxes(ax1, ax2)
  return checkaxes(Bool, ax1, ax2) ||
         throw(DimensionMismatch(lazy"$ax1 does not match $ax2"))
end
function checkaxes(::Type{Bool}, axes1, axes2)
  return length(axes1) == length(axes2) && all(space_isequal.(axes1, axes2))
end
