# This file defines struct FusionTensor and constructors

using BlockSparseArrays: block_stored_indices

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
    @assert keytype(trees_block_mapping) <: Tuple{<:FusionTree{S},<:FusionTree{S}}
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

# GradedUnitRanges interface
GradedUnitRanges.sector_type(ft::FusionTensor) = sector_type(matrix_row_axis(ft))
function GradedUnitRanges.findblock(ft::FusionTensor, f1::FusionTree, f2::FusionTree)
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

function sanitize_axes(raw_legs)
  legs = unify_sector_type(raw_legs)
  @assert all(check_unique_blocklabels.(legs))
  return legs
end

function unify_sector_type(legs::Tuple{Vararg{AbstractGradedUnitRange{LA}}}) where {LA}  # nothing to do
  return legs
end

check_unique_blocklabels(g) = length(unique(blocklabels(g))) == blocklength(g)

# TODO move this to SymmetrySectors or GradedUnitRanges
# merge with SymmetrySectors.map_blocklabels
function find_common_sector_type(sector_or_axes_enum)
  # fuse trivial sectors to produce unified type
  # avoid depending on SymmetrySectors internals
  return label_type(fusion_product(trivial.(sector_or_axes_enum)...))
end

function unify_sector_type(legs::Tuple{Vararg{AbstractGradedUnitRange}})
  T = find_common_sector_type(legs)
  unified_legs = map(g -> unify_sector_type(T, g), legs)
  return unified_legs
end

function unify_sector_type(T::Type{<:SectorProduct}, g::AbstractGradedUnitRange)
  # fuse with trivial to insert all missing arguments inside each GradedAxis
  # avoid depending on SymmetrySectors internals
  glabels = map(s -> only(blocklabels(fusion_product(trivial(T), s))), blocklabels(g))
  # use labelled_blocks to preserve GradedUnitRange
  unified_g = labelled_blocks(unlabel_blocks(g), glabels)
  return isdual(g) ? flip(unified_g) : unified_g
end

function FusionTensor(
  mat::AbstractBlockSparseMatrix,
  codomain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
  domain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
)
  ft = FusionTensor(eltype(mat), codomain_legs, domain_legs)
  for b in block_stored_indices(mat)
    @assert last(b) in block_stored_indices(data_matrix(ft))
    data_matrix(ft)[last(b)] = mat[last(b)]
  end
  return ft
end

# empty matrix
function FusionTensor(
  data_type::Type,
  codomain_legs_raw::Tuple{Vararg{AbstractGradedUnitRange}},
  domain_legs_raw::Tuple{Vararg{AbstractGradedUnitRange}},
)
  legs = sanitize_axes((codomain_legs_raw..., domain_legs_raw...))
  S = sector_type(first(legs))
  codomain_legs = legs[begin:length(codomain_legs_raw)]
  domain_legs = legs[(length(codomain_legs_raw) + 1):end]
  codomain_fused_axes = FusedAxes{S}(codomain_legs)
  domain_fused_axes = FusedAxes{S}(dual.(domain_legs))
  mat = initialize_data_matrix(data_type, codomain_fused_axes, domain_fused_axes)
  tree_to_block_mapping = intersect_sectors(codomain_fused_axes, domain_fused_axes)
  return FusionTensor(mat, codomain_legs, domain_legs, tree_to_block_mapping)
end

function FusionTensor(data_type::Type, ::Tuple{}, ::Tuple{})
  codomain_fused_axes = FusedAxes{TrivialSector}(())
  domain_fused_axes = FusedAxes{TrivialSector}(())
  mat = initialize_data_matrix(data_type, codomain_fused_axes, domain_fused_axes)
  tree_to_block_mapping = intersect_sectors(codomain_fused_axes, domain_fused_axes)
  return FusionTensor(mat, (), (), tree_to_block_mapping)
end

# init data_matrix
function initialize_data_matrix(
  data_type::Type{<:Number}, codomain_fused_axes::FusedAxes, domain_fused_axes::FusedAxes
)
  # fusion trees have Float64 eltype: need compatible type
  promoted = promote_type(data_type, Float64)
  mat_row_axis = fused_axis(codomain_fused_axes)
  mat_col_axis = dual(fused_axis(domain_fused_axes))
  mat = BlockSparseArray{promoted}(mat_row_axis, mat_col_axis)
  initialize_allowed_sectors!(mat)
  return mat
end

function initialize_allowed_sectors!(mat::AbstractBlockMatrix)
  row_sectors = blocklabels(axes(mat, 1))
  col_sectors = blocklabels(dual(axes(mat, 2)))
  row_block_indices = findall(in(col_sectors), row_sectors)
  col_block_indices = findall(in(row_sectors), col_sectors)
  for rc in zip(row_block_indices, col_block_indices)
    mat[Block(rc)] = mat[Block(rc)]
  end
end

# TODO move to tests
function check_data_matrix_axes(
  mat::BlockSparseMatrix, codomain_legs::Tuple, domain_legs::Tuple
)
  ft0 = FusionTensor(Float64, codomain_legs, domain_legs)
  @assert space_isequal(matrix_row_axis(ft0), axes(mat, 1))
  @assert space_isequal(matrix_column_axis(ft0), axes(mat, 2))
end

function check_data_matrix_axes(mat::Adjoint, codomain_legs::Tuple, domain_legs::Tuple)
  return check_data_matrix_axes(adjoint(mat), dual.(domain_legs), dual.(codomain_legs))
end

# TODO move to tests
function check_sanity(ft::FusionTensor)
  nca = ndims_domain(ft)
  @assert nca == length(domain_axes(ft)) "ndims_domain does not match domain_axes"
  @assert nca <= ndims(ft) "invalid ndims_domain"

  nda = ndims_codomain(ft)
  @assert nda == length(codomain_axes(ft)) "ndims_codomain does not match codomain_axes"
  @assert nda <= ndims(ft) "invalid ndims_codomain"
  @assert nda + nca == ndims(ft) "invalid ndims"

  @assert length(axes(ft)) == ndims(ft) "ndims does not match axes"
  @assert matching_axes(axes(ft)[begin:nda], codomain_axes(ft)) "axes do not match codomain_axes"
  @assert matching_axes(axes(ft)[(nda + 1):end], domain_axes(ft)) "axes do not match domain_axes"

  m = data_matrix(ft)
  @assert ndims(m) == 2 "invalid data_matrix ndims"
  row_axis = matrix_row_axis(ft)
  column_axis = matrix_column_axis(ft)
  @assert row_axis === axes(m, 1) "invalid row_axis"
  @assert column_axis === axes(m, 2) "invalid column_axis"
  check_data_matrix_axes(data_matrix(ft), codomain_axes(ft), domain_axes(ft))

  for b in block_stored_indices(m)
    it = Int.(Tuple(b))
    @assert blocklabels(row_axis)[it[1]] == blocklabels(dual(column_axis))[it[2]] "forbidden block"
  end
  return nothing
end

matching_dual(axes1::Tuple, axes2::Tuple) = matching_axes(dual.(axes1), axes2)
matching_axes(axes1::Tuple, axes2::Tuple) = false
function matching_axes(axes1::T, axes2::T) where {T<:Tuple}
  return all(space_isequal.(axes1, axes2))
end