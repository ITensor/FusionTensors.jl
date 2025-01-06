# This file defines struct FusionTensor and constructors

using BlockArrays: AbstractBlockMatrix, BlockArrays, blocklength, findblock

using BlockSparseArrays: AbstractBlockSparseMatrix, BlockSparseArray, eachblockstoredindex
using GradedUnitRanges:
  AbstractGradedUnitRange,
  blocklabels,
  dual,
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
  # TBD replace codomain_legs with FusedAxes(codomain_legs)?
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
  ::Type{<:FusionTensor{<:Any,<:Any,CoDomainAxes}}
) where {CoDomainAxes}
  return sector_type(fieldtype(CoDomainAxes, 1))
end
function GradedUnitRanges.sector_type(
  ::Type{<:FusionTensor{<:Any,<:Any,Tuple{},DomainAxes}}
) where {DomainAxes}
  return sector_type(fieldtype(DomainAxes, 1))
end
function GradedUnitRanges.sector_type(::Type{<:FusionTensor{<:Any,0}})
  return TrivialSector
end

# BlockArrays interface
function BlockArrays.findblock(ft::FusionTensor, f1::SectorFusionTree, f2::SectorFusionTree)
  # find outer block corresponding to fusion trees
  @assert ndims_codomain(ft) == length(f1)
  @assert ndims_domain(ft) == length(f2)
  @assert sector_type(ft) == sector_type(f1)
  @assert sector_type(ft) == sector_type(f2)
  b1 = ntuple(
    i -> findfirst(==(leaves(f1)[i]), blocklabels(codomain_axes(ft)[i])), ndims_codomain(ft)
  )
  b2 = ntuple(
    i -> findfirst(==(leaves(f2)[i]), blocklabels(dual(domain_axes(ft)[i]))),
    ndims_domain(ft),
  )
  return Block(b1..., b2...)
end

sanitize_axes(::Tuple{}) = ()
function sanitize_axes(raw_legs::Tuple{Vararg{AbstractGradedUnitRange}})
  legs = promote_sectors(typeof(first(raw_legs)), raw_legs)
  @assert all(check_unique_blocklabels.(legs))
  return legs
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
function fusiontensor(
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
  legs = sanitize_axes((codomain_legs_raw..., domain_legs_raw...))
  S = sector_type(first(legs))
  codomain_legs = legs[begin:length(codomain_legs_raw)]
  domain_legs = legs[(length(codomain_legs_raw) + 1):end]
  codomain_fused_axes = FusedAxes{S}(codomain_legs)
  domain_fused_axes = FusedAxes{S}(dual.(domain_legs))
  mat = initialize_data_matrix(elt, codomain_fused_axes, domain_fused_axes)
  tree_to_block_mapping = intersect_sectors(codomain_fused_axes, domain_fused_axes)
  return FusionTensor(mat, codomain_legs, domain_legs, tree_to_block_mapping)
end

function FusionTensor(elt::Type, ::Tuple{}, ::Tuple{})
  codomain_fused_axes = FusedAxes{TrivialSector}(())
  domain_fused_axes = FusedAxes{TrivialSector}(())
  mat = initialize_data_matrix(elt, codomain_fused_axes, domain_fused_axes)
  tree_to_block_mapping = intersect_sectors(codomain_fused_axes, domain_fused_axes)
  return FusionTensor(mat, (), (), tree_to_block_mapping)
end

function initialize_data_matrix(
  elt::Type{<:Number}, codomain_fused_axes::FusedAxes, domain_fused_axes::FusedAxes
)
  # fusion trees have Float64 eltype: need compatible type
  promoted = promote_type(elt, Float64)
  mat_row_axis = fused_axis(codomain_fused_axes)
  mat_col_axis = dual(fused_axis(domain_fused_axes))
  mat = BlockSparseArray{promoted}(mat_row_axis, mat_col_axis)
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
