# This file defines struct FusionTensor and constructors

using BlockArrays: AbstractBlockMatrix, BlockArrays, BlockIndexRange, blocklength, findblock

using BlockSparseArrays:
  AbstractBlockSparseMatrix, BlockSparseArray, eachblockstoredindex, to_block_indices
using GradedArrays:
  AbstractGradedUnitRange,
  SectorProduct,
  TrivialSector,
  dual,
  flip,
  flip_dual,
  gradedrange,
  isdual,
  map_sectors,
  sector_multiplicity,
  sector_type,
  sectormergesort,
  sectors,
  space_isequal
using LinearAlgebra: UniformScaling
using TensorAlgebra: BlockedTuple, trivial_axis, tuplemortar
using TensorProducts: tensor_product
using TypeParameterAccessors: type_parameters

# =======================================  Misc  ===========================================

# TBD move to GradedArrays? rename findfirst_sector?
function find_sector_block(s::AbstractSector, g::AbstractGradedUnitRange)
  return findfirst(==(s), sectors(flip_dual(g)))
end

# TBD move to GradedArrays?
function checkaxes(::Type{Bool}, axes1, axes2)
  return length(axes1) == length(axes2) && all(space_isequal.(axes1, axes2))
end

# TBD move to GradedArrays?
checkaxes_dual(axes1, axes2) = checkaxes(axes1, dual.(axes2))
function checkaxes(ax1, ax2)
  return checkaxes(Bool, ax1, ax2) ||
         throw(DimensionMismatch(lazy"$ax1 does not match $ax2"))
end

function to_blockindexrange(b1::BlockIndexRange{1}, b2::BlockIndexRange{1})
  t = (b1, b2)
  return Block(Block.(t))[to_block_indices.(t)...]
end

function flip_domain(nonflipped_col_axis, nonflipped_trees_to_ranges)
  col_axis = dual(nonflipped_col_axis)
  domain_trees_to_ranges_mapping = Dict(
    map(((tree, v),) -> flip(tree) => v, collect(nonflipped_trees_to_ranges))
  )
  return col_axis, domain_trees_to_ranges_mapping
end

function fuse_axes(::Type{S}, ::Tuple{}) where {S<:AbstractSector}
  fused_axis = trivial_axis(S)
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
  block_sectors = only.(sectors.(flip_dual.(it)))
  block_mult = prod(sector_multiplicity.(it))
  return build_trees(block_sectors, isdual.(it)) .=> block_mult
end

function compute_inner_ranges(fusion_trees_mult)
  fused_leg = sectormergesort(
    gradedrange(root_sector.(first.(fusion_trees_mult)) .=> last.(fusion_trees_mult))
  )
  range_mapping = Dict{type_parameters(eltype(fusion_trees_mult), 1),typeof(Block(1)[1:1])}()
  fused_sectors = sectors(fused_leg)
  shifts = ones(Int, blocklength(fused_leg))
  for (f, m) in fusion_trees_mult
    s = root_sector(f)
    i = findfirst(==(s), fused_sectors)
    range_mapping[f] = Block(i)[shifts[i]:(shifts[i] + m - 1)]
    shifts[i] += m
  end
  return fused_leg, range_mapping
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
  codomain_axis::AbstractGradedUnitRange,
  domain_axis::AbstractGradedUnitRange,
)
  @assert sector_type(codomain_axis) == sector_type(domain_axis)
  # non-abelian fusion trees have float eltype: need compatible type
  promoted = promote_type(elt, fusiontree_eltype(sector_type(domain_axis)))
  mat = BlockSparseArray{promoted}(
    undef,
    blockedrange(sector_multiplicities(codomain_axis)),
    blockedrange(sector_multiplicities(domain_axis)),
  )
  row_sectors = sectors(codomain_axis)
  col_sectors = sectors(domain_axis)
  row_block_indices = findall(in(col_sectors), row_sectors)
  col_block_indices = findall(in(row_sectors), col_sectors)
  for rc in zip(row_block_indices, col_block_indices)
    mat[Block(rc)] = mat[Block(rc)]
  end
  return mat
end

# ====================================  Definitions  =======================================

struct FusionTensor{T,N,Axes<:FusionTensorAxes,Mat<:AbstractMatrix{T},Mapping} <:
       AbstractArray{T,N}
  data_matrix::Mat
  axes::Axes
  trees_block_mapping::Mapping

  # inner constructor to impose constraints on types
  function FusionTensor{T,N,Axes,Mat,Mapping}(
    mat, legs, trees_block_mapping
  ) where {T,N,Axes,Mat,Mapping}
    S = sector_type(legs)
    @assert keytype(trees_block_mapping) <:
      Tuple{<:SectorFusionTree{S},<:SectorFusionTree{S}}
    return new{T,N,Axes,Mat,Mapping}(mat, legs, trees_block_mapping)
  end
end

# =====================================  Accessors  ========================================

data_matrix(ft::FusionTensor) = ft.data_matrix
trees_block_mapping(ft::FusionTensor) = ft.trees_block_mapping

# ====================================  Constructors  ======================================

function FusionTensor(
  mat::AbstractMatrix,
  legs::FusionTensorAxes,
  trees_block_mapping::Dict{<:Tuple{<:SectorFusionTree,<:SectorFusionTree}},
)
  return FusionTensor{
    eltype(mat),length(legs),typeof(legs),typeof(mat),typeof(trees_block_mapping)
  }(
    mat, legs, trees_block_mapping
  )
end

# empty matrix
function FusionTensor(elt::Type, legs::FusionTensorAxes)
  S = sector_type(legs)
  row_axis, codomain_trees_to_ranges = fuse_axes(S, codomain(legs))
  col_axis, domain_trees_to_ranges = flip_domain(fuse_axes(S, dual.(domain(legs)))...)

  mat = initialize_data_matrix(elt, row_axis, col_axis)
  tree_to_block_mapping = intersect_codomain_domain(
    codomain_trees_to_ranges, domain_trees_to_ranges
  )
  return FusionTensor(mat, legs, tree_to_block_mapping)
end

#constructor from precomputed data_matrix
function FusionTensor(mat::AbstractMatrix, legs::FusionTensorAxes)
  # init with empty data_matrix to construct trees_block_mapping
  ft = FusionTensor(eltype(mat), legs)
  for b in eachblockstoredindex(mat)
    b in eachblockstoredindex(data_matrix(ft)) ||
      throw(ArgumentError("matrix block $b is not allowed"))
    data_matrix(ft)[b] = mat[b]
  end
  return ft
end

FusionTensor(x, legs::BlockedTuple{2}) = FusionTensor(x, FusionTensorAxes(legs))

# constructor from split axes
function FusionTensor(
  x,
  codomain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
  domain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
)
  return FusionTensor(x, tuplemortar((codomain_legs, domain_legs)))
end

# specific constructors
Base.zeros(::Type{T}, fta::FusionTensorAxes) where {T} = FusionTensor(T, fta)
Base.zeros(fta::FusionTensorAxes) = zeros(Float64, fta)

function Base.randn(::Type{T}, fta::FusionTensorAxes) where {T}
  ft = FusionTensor(T, fta)
  for m in eachstoredblock(data_matrix(ft))
    m = randn!(m)
  end
  return ft
end
Base.randn(fta::FusionTensorAxes) = randn(Float64, fta)

function FusionTensor(
  ::UniformScaling, codomain_legs::Tuple{Vararg{AbstractGradedUnitRange}}
)
  fta = FusionTensorAxes(codomain_legs, dual.(codomain_legs))
  ft = FusionTensor(Float64, fta)
  for m in eachstoredblock(data_matrix(ft))
    m .= LinearAlgebra.I(size(m, 1))
  end
  return ft
end

# ================================  BlockArrays interface  =================================

function BlockArrays.findblock(ft::FusionTensor, f1::SectorFusionTree, f2::SectorFusionTree)
  # find outer block corresponding to fusion trees
  @assert typeof((f1, f2)) === keytype(trees_block_mapping(ft))
  b1 = find_sector_block.(leaves(f1), codomain_axes(ft))
  b2 = find_sector_block.(leaves(f2), domain_axes(ft))
  return Block(b1..., b2...)
end

# ==============================  GradedArrays interface  ==================================

function GradedArrays.sector_type(::Type{FT}) where {FT<:FusionTensor}
  return sector_type(type_parameters(FT, 3))
end

# ==============================  FusionTensor interface  ==================================

# misc access
codomain_axes(ft::FusionTensor) = codomain(axes(ft))

domain_axes(ft::FusionTensor) = domain(axes(ft))

codomain_axis(ft::FusionTensor) = fused_codomain(axes(ft))

domain_axis(ft::FusionTensor) = fused_domain(axes(ft))

ndims_codomain(ft::FusionTensor) = length_codomain(axes(ft))

ndims_domain(ft::FusionTensor) = length_domain(axes(ft))

function charge_block_size(ft::FusionTensor, f1::SectorFusionTree, f2::SectorFusionTree)
  b = Tuple(findblock(ft, f1, f2))
  return ntuple(i -> sector_multiplicity(axes(ft, i)[b[i]]), ndims(ft))
end
