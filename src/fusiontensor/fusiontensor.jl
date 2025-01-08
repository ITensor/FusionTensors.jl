# This file defines struct FusionTensor and constructors

using BlockArrays: AbstractBlockMatrix, BlockArrays, BlockIndexRange, blocklength, findblock

using BlockSparseArrays:
  AbstractBlockSparseMatrix, BlockSparseArray, eachblockstoredindex, to_block_indices
using GradedUnitRanges:
  AbstractGradedUnitRange,
  blocklabels,
  blockmergesort,
  dual,
  gradedrange,
  isdual,
  map_blocklabels,
  sector_type,
  space_isequal
using SymmetrySectors: SectorProduct, TrivialSector

struct FusionTensor{T,N,CoDomainAxes,DomainAxes,Mat,Mapping} <: AbstractArray{T,N}
  data_matrix::Mat
  codomain_axes::CoDomainAxes
  domain_axes::DomainAxes
  trees_block_mapping::Mapping

  # inner constructor to impose constraints on types
  function FusionTensor(
    mat::AbstractMatrix,
    codomain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
    domain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
    trees_block_mapping::Dict,
  )
    S = sector_type(axes(mat, 1))
    @assert sector_type(axes(mat, 2)) === S
    @assert keytype(trees_block_mapping) <:
      Tuple{<:SectorFusionTree{S},<:SectorFusionTree{S}}
    @assert all(sector_type.(codomain_legs) .=== S)
    @assert all(sector_type.(domain_legs) .=== S)
    return new{
      eltype(mat),
      length(codomain_legs) + length(domain_legs),
      typeof(codomain_legs),
      typeof(domain_legs),
      typeof(mat),
      typeof(trees_block_mapping),
    }(
      mat, codomain_legs, domain_legs, trees_block_mapping
    )
  end
end

# getters
data_matrix(ft::FusionTensor) = ft.data_matrix
codomain_axes(ft::FusionTensor) = ft.codomain_axes
domain_axes(ft::FusionTensor) = ft.domain_axes
trees_block_mapping(ft::FusionTensor) = ft.trees_block_mapping

# misc access
ndims_codomain(ft::FusionTensor) = length(codomain_axes(ft))
ndims_domain(ft::FusionTensor) = length(domain_axes(ft))

matrix_size(ft::FusionTensor) = quantum_dimension.(axes(data_matrix(ft)))
matrix_row_axis(ft::FusionTensor) = first(axes(data_matrix(ft)))
matrix_column_axis(ft::FusionTensor) = last(axes(data_matrix(ft)))
function charge_block_size(ft::FusionTensor, f1::SectorFusionTree, f2::SectorFusionTree)
  b = Tuple(findblock(ft, f1, f2))
  return ntuple(i -> Int(length(axes(ft)[i][b[i]])), ndims(ft))
end

# GradedUnitRanges interface
function GradedUnitRanges.sector_type(
  ::Type{<:FusionTensor{<:Any,<:Any,<:Any,<:Any,<:Any,<:Dict{<:Tuple{<:Any,F}}}}
) where {F}
  return sector_type(F)
end

# BlockArrays interface
function BlockArrays.findblock(ft::FusionTensor, f1::SectorFusionTree, f2::SectorFusionTree)
  # find outer block corresponding to fusion trees
  @assert typeof((f1, f2)) === keytype(trees_block_mapping(ft))
  b1 = find_sector_block.(leaves(f1), codomain_axes(ft))
  b2 = find_sector_block.(leaves(f2), dual.(domain_axes(ft)))
  return Block(b1..., b2...)
end
# TBD move to GradedUnitRanges? rename findfirst_sector?
function find_sector_block(s::AbstractSector, l::AbstractGradedUnitRange)
  return findfirst(==(s), blocklabels(l))
end

function sanitize_axes(raw_legs::Tuple{Vararg{AbstractGradedUnitRange}})
  legs = promote_sectors(typeof(first(raw_legs)), raw_legs)
  @assert all(check_unique_blocklabels.(legs))
  return legs
end
sanitize_axes(::Tuple{}, ::Tuple{}) = TrivialSector, (), ()
function sanitize_axes(
  codomain_legs_raw::Tuple{Vararg{AbstractGradedUnitRange}},
  domain_legs_raw::Tuple{Vararg{AbstractGradedUnitRange}},
)
  legs = sanitize_axes((codomain_legs_raw..., domain_legs_raw...))
  S = sector_type(first(legs))
  codomain_legs = legs[begin:length(codomain_legs_raw)]
  domain_legs = legs[(length(codomain_legs_raw) + 1):end]
  return S, domain_legs, codomain_legs
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
  unified_legs = map_blocklabels.(s -> only(blocklabels(fusion_product(s0, s))), legs)
  return unified_legs
end

function promote_sector_type(legs)
  # fuse trivial sectors to produce unified type
  # avoid depending on SymmetrySectors internals
  return sector_type(fusion_product(trivial.(legs)...))
end

# initialize with already computed data_matrix
function FusionTensor(
  mat::AbstractMatrix,
  codomain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
  domain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
)
  # init with empty data_matrix to construct trees_block_mapping
  ft = FusionTensor(eltype(mat), codomain_legs, domain_legs)
  @assert space_isequal(matrix_row_axis(ft), axes(mat, 1))
  @assert space_isequal(matrix_column_axis(ft), axes(mat, 2))
  for b in eachblockstoredindex(mat)
    @assert b in eachblockstoredindex(data_matrix(ft))  # check matrix block is allowed
    data_matrix(ft)[b] = mat[b]
  end
  return ft
end

# empty matrix
function FusionTensor(
  elt::Type,
  codomain_legs_raw::Tuple{Vararg{AbstractGradedUnitRange}},
  domain_legs_raw::Tuple{Vararg{AbstractGradedUnitRange}},
)
  S, domain_legs, codomain_legs = sanitize_axes(codomain_legs_raw, domain_legs_raw)

  row_axis, codomain_trees_to_ranges_mapping = fuse_axes(S, codomain_legs)
  nondual_col_axis, domain_trees_to_ranges_mapping = fuse_axes(S, dual.(domain_legs))

  mat = initialize_data_matrix(elt, row_axis, nondual_col_axis)
  tree_to_block_mapping = intersect_codomain_domain(
    codomain_trees_to_ranges_mapping, domain_trees_to_ranges_mapping
  )
  return FusionTensor(mat, codomain_legs, domain_legs, tree_to_block_mapping)
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
  tree_arrows = isdual.(outer_legs)
  return mapreduce(vcat, CartesianIndices(blocklength.(outer_legs))) do it
    block_sectors = map((g, i) -> blocklabels(g)[i], outer_legs, Tuple(it))
    block_mult = mapreduce((g, i) -> blocklengths(g)[i], *, outer_legs, Tuple(it); init=1)
    return build_trees(block_sectors, tree_arrows) .=> block_mult
  end
end

function compute_inner_ranges(
  fusion_trees_mult::AbstractVector{<:Pair{<:SectorFusionTree,<:Integer}}
)
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
        t -> root_sector(first(t[1])) == root_sector(first(t[2])),
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
  nondual_col_axis::AbstractGradedUnitRange,
)
  # non-abelian fusion trees have float eltype: need compatible type
  promoted = promote_type(elt, fusiontree_eltype(sector_type(mat_row_axis)))
  mat = BlockSparseArray{promoted}(mat_row_axis, dual(nondual_col_axis))
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
