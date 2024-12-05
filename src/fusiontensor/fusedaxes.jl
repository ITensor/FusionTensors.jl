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

function FusedAxes{S}(
  outer_legs::Tuple{Vararg{AbstractGradedUnitRange}}
) where {S<:AbstractSector}
  fusion_trees_mult = fusion_trees_external_multiplicites(outer_legs)

  fused_leg, range_mapping = compute_inner_ranges(fusion_trees_mult)
  return FusedAxes(outer_legs, fused_leg, range_mapping)
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

function to_blockindexrange(b1::BlockIndexRange{1}, b2::BlockIndexRange{1})
  t = (b1, b2)
  return Block(Block.(t))[BlockSparseArrays.to_block_indices.(t)...]
end

# TBD choose one
function intersect_sectors(left::FusedAxes, right::FusedAxes)
  shared_sectors = intersect(blocklabels(left), blocklabels(right))
  blockindexrange_vec = mapreduce(vcat, shared_sectors) do s
    codomain_trees = filter(f -> root_sector(f) == s, keys(trees_to_ranges_mapping(left)))
    domain_trees = filter(f -> root_sector(f) == s, keys(trees_to_ranges_mapping(right)))
    return vec(
      collect(
        (f1, f2) => to_blockindexrange(
          trees_to_ranges_mapping(left)[f1], trees_to_ranges_mapping(left)[f2]
        ) for f1 in codomain_trees, f2 in domain_trees
      ),
    )
  end
  return Dict(blockindexrange_vec)
end

function intersect_sectors2(left::FusedAxes, right::FusedAxes)
  return Dict(
    map(
      t -> first.(t) => to_blockindexrange(last.(t)...),
      Iterators.filter(
        t -> root_sector(first(t[1])) == root_sector(first(t[2])),
        Iterators.product(trees_to_ranges_mapping(left), trees_to_ranges_mapping(right)),
      ),
    ),
  )
end
