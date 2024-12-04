# This file defines helper functions to access FusionTensor internal structures

struct FusedAxes{A,B,C}
  outer_axes::A
  fused_axis::B
  trees_to_ranges_mapping::C

  function FusedAxes(
    outer_legs::NTuple{N,AbstractGradedUnitRange{LA}},
    fused_axis::AbstractGradedUnitRange{LA},
    trees_to_ranges_mapping::Dict{<:FusionTree{<:AbstractSector,N}},
  ) where {N,LA}
    return new{typeof(outer_legs),typeof(fused_axis),typeof(trees_to_ranges_mapping)}(
      outer_legs, fused_axis, trees_to_ranges_mapping
    )
  end
end

# getters
fused_axis(fa::FusedAxes) = fa.fused_axis
fusion_trees(fa::FusedAxes) = keys(trees_to_ranges_mapping(fa))
trees_to_ranges_mapping(fa::FusedAxes) = fa.trees_to_ranges_mapping

# Base interface
Base.axes(fa::FusedAxes) = fa.outer_axes
Base.ndims(fa::FusedAxes) = length(axes(fa))

# GradedUnitRanges interface
GradedUnitRanges.blocklabels(fa::FusedAxes) = blocklabels(fused_axis(fa))

# constructors
function FusedAxes{S}(::Tuple{}) where {S<:AbstractSector}
  fused_axis = gradedrange([trivial(S) => 1])
  trees_to_ranges_mapping = Dict([FusionTree{S}() => Block(1)[1:1]])
  return FusedAxes((), fused_axis, trees_to_ranges_mapping)
end

function fusion_trees_external_multiplicites(
  outer_legs::Tuple{Vararg{AbstractGradedUnitRange}}
)
  N = length(outer_legs)
  tree_arrows = isdual.(outer_legs)
  return mapreduce(vcat, CartesianIndices(blocklength.(outer_legs))) do it
    block_sectors = ntuple(i -> blocklabels(outer_legs[i])[it[i]], N)
    block_mult = prod(ntuple(i -> blocklengths(outer_legs[i])[it[i]], N))
    return build_trees(block_sectors, tree_arrows) .=> block_mult
  end
end

function FusedAxes{S}(
  outer_legs::Tuple{Vararg{AbstractGradedUnitRange}}
) where {S<:AbstractSector}
  fusion_trees_mult = fusion_trees_external_multiplicites(outer_legs)

  fused_leg, range_mapping = compute_inner_ranges(fusion_trees_mult)
  return FusedAxes(outer_legs, fused_leg, range_mapping)
end

function compute_inner_ranges(
  fusion_trees_mult::AbstractVector{<:Pair{<:FusionTree,<:Integer}}
)
  fused_leg = blockmergesort(
    gradedrange(root_sector.(first.(fusion_trees_mult)) .=> last.(fusion_trees_mult))
  )
  range_mapping = Dict{typeof(first(first(fusion_trees_mult))),typeof(Block(1)[1:1])}()
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

function Base.intersect(left::FusedAxes, right::FusedAxes)
  left_labels = blocklabels(left)
  right_labels = blocklabels(right)
  return find_shared_indices(left_labels, right_labels)
end

function allowed_outer_blocks_sectors(
  left::FusedAxes, right::FusedAxes, shared_indices::AbstractVector
)
  left_labels = blocklabels(left)
  left_indices = first.(shared_indices)
  right_indices = last.(shared_indices)
  @assert left_labels[left_indices] == blocklabels(right)[right_indices]
  reduced_left = .!isempty.(index_matrix(left)[:, left_indices])
  reduced_right = .!isempty.(index_matrix(right)[:, right_indices])

  block_sectors = Dict{NTuple{ndims(left) + ndims(right),Int},Vector{eltype(left_labels)}}()
  for i in axes(reduced_left, 1), j in axes(reduced_right, 1)
    intersection = findall(>(0), reduced_left[i, :] .* reduced_right[j, :])
    isempty(intersection) && continue
    full_block = (unravel_index(i, left)..., unravel_index(j, right)...)
    block_sectors[full_block] = left_labels[left_indices[intersection]]
  end
  return block_sectors
end
