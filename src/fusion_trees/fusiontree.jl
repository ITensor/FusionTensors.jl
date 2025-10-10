# This file defines fusion trees for any abelian or non-abelian group

# TBD
# compatibility with TensorKit conventions?

using GradedArrays:
    GradedArrays,
    AbstractGradedUnitRange,
    SectorProductRange,
    SectorRange,
    ×,
    arguments,
    flip,
    flip_dual,
    isdual,
    nsymbol,
    sector_multiplicities,
    sector_type,
    sectorproduct,
    sectors,
    to_gradedrange,
    trivial
using TensorAlgebra: flatten_tuples
using TensorProducts: ⊗

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
# The interface uses AbstractGradedArrays as input for interface simplicity
# however only sectors are used and blocklengths are never read.

struct SectorFusionTree{S, N, M}
    leaves::NTuple{N, S}  # TBD rename outer_sectors or leave_sectors?
    arrows::NTuple{N, Bool}
    root_sector::S
    branch_sectors::NTuple{M, S}  # M = N-1
    outer_multiplicity_indices::NTuple{M, Int}  # M = N-1

    # TBD could have branch_sectors with length N-2
    # currently first(branch_sectors) == first(leaves)
    # redundant but allows for simpler, generic grow_tree code

    function SectorFusionTree(
            leaves, arrows, root_sector, branch_sectors, outer_multiplicity_indices
        )
        N = length(leaves)
        @assert length(branch_sectors) == max(0, N - 1)
        @assert length(outer_multiplicity_indices) == max(0, N - 1)
        return new{typeof(root_sector), length(leaves), length(branch_sectors)}(
            leaves, arrows, root_sector, branch_sectors, outer_multiplicity_indices
        )
    end
end

# getters
arrows(f::SectorFusionTree) = f.arrows
leaves(f::SectorFusionTree) = f.leaves
root_sector(f::SectorFusionTree) = f.root_sector
branch_sectors(f::SectorFusionTree) = f.branch_sectors
outer_multiplicity_indices(f::SectorFusionTree) = f.outer_multiplicity_indices

# Base interface
Base.convert(T::Type{<:Array}, f::SectorFusionTree) = convert(T, to_array(f))

function Base.isless(f1::SectorFusionTree, f2::SectorFusionTree)
    return isless(leaves(f1), leaves(f2)) ||
        isless(arrows(f1), arrows(f2)) ||
        isless(root_sector(f1), root_sector(f2)) ||
        isless(branch_sectors(f1), branch_sectors(f2)) ||
        isless(outer_multiplicity_indices(f1), outer_multiplicity_indices(f2))
end

Base.length(::SectorFusionTree{<:Any, N}) where {N} = N

# GradedArrays interface
GradedArrays.sector_type(::Type{<:SectorFusionTree{S}}) where {S} = S
function GradedArrays.flip(f::SectorFusionTree)
    return SectorFusionTree(
        dual.(leaves(f)),
        .!arrows(f),
        dual(root_sector(f)),
        dual.(branch_sectors(f)),
        outer_multiplicity_indices(f),
    )
end

function GradedArrays.:×(f1::SectorFusionTree, f2::SectorFusionTree)
    @assert arrows(f1) == arrows(f2)
    product_leaves = .×(leaves(f1), leaves(f2))
    product_root_sector = root_sector(f1) × root_sector(f2)
    product_branch_sectors = .×(branch_sectors(f1), branch_sectors(f2))
    product_outer_multiplicity_indices = outer_multiplicity_kron.(
        Base.tail(leaves(f1)),
        branch_sectors(f1),
        (Base.tail(branch_sectors(f1))..., root_sector(f1)),
        outer_multiplicity_indices(f1),
        outer_multiplicity_indices(f2),
    )
    return SectorFusionTree(
        product_leaves,
        arrows(f1),
        product_root_sector,
        product_branch_sectors,
        product_outer_multiplicity_indices,
    )
end

function GradedArrays.arguments(f::SectorFusionTree{<:SectorProductRange})
    transposed_indices = outer_multiplicity_split.(
        Base.tail(leaves(f)),
        branch_sectors(f),
        (Base.tail(branch_sectors(f))..., root_sector(f)),
        outer_multiplicity_indices(f),
    )
    arguments_root = arguments(root_sector(f))
    arguments_leaves = arguments.(leaves(f))
    arguments_branch_sectors = arguments.(branch_sectors(f))
    # TODO way to avoid explicit ntuple?
    # works fine for Tuple and NamedTuple SectorProductRange
    return ntuple(
        i -> SectorFusionTree(
            getindex.(arguments_leaves, i),
            arrows(f),
            arguments_root[i],
            getindex.(arguments_branch_sectors, i),
            getindex.(transposed_indices, i),
        ),
        length(arguments_root),
    )
end

function GradedArrays.arguments(f::SectorFusionTree{<:SectorProductRange, 0})
    return map(arg -> SectorFusionTree((), (), arg, (), ()), arguments(root_sector(f)))
end

function GradedArrays.arguments(f::SectorFusionTree{<:SectorProductRange, 1})
    arguments_root = arguments(root_sector(f))
    arguments_leave = arguments(only(leaves(f)))
    # use map(keys) to stay agnostic with respect to SectorProductRange implementation
    return map(keys(arguments_root)) do k
        return SectorFusionTree((arguments_leave[k],), arrows(f), arguments_root[k], (), ())
    end
end

# TBD change type depending on AbelianStyle?
fusiontree_eltype(::Type{<:SectorRange}) = Float64

# constructors
function build_trees(legs::Vararg{AbstractGradedUnitRange})
    # construct all authorized trees for each outer block in legs
    tree_arrows = isdual.(legs)
    return mapreduce(vcat, Iterators.product(sectors.(flip_dual.(legs))...)) do it
        return build_trees(it, tree_arrows)
    end
end

function build_trees(
        sectors_to_fuse::NTuple{N, <:SectorRange}, arrows_to_fuse::NTuple{N, Bool}
    ) where {N}
    # construct all authorized trees with fixed outer sectors
    trees = [SectorFusionTree(first(sectors_to_fuse), first(arrows_to_fuse))]
    return recursive_build_trees(trees, Base.tail(sectors_to_fuse), Base.tail(arrows_to_fuse))
end

#
# =====================================  Internals  ========================================
#

# --------------- SectorProductRange helper functions  ---------------
function outer_multiplicity_kron(
        sec1, sec2, fused, outer_multiplicity1, outer_multiplicity2
    )
    n = Int(nsymbol(sec1, sec2, fused))
    linear_inds = LinearIndices((n, outer_multiplicity2))
    return linear_inds[outer_multiplicity1, outer_multiplicity2]
end

function outer_multiplicity_split(
        sec1::S, sec2::S, fused::S, outer_mult_index::Integer
    ) where {S <: SectorProductRange}
    args1 = arguments(sec1)
    args2 = arguments(sec2)
    args12 = arguments(fused)
    nsymbols = Tuple(map(nsymbol, args1, args2, args12))  # CartesianIndices requires explicit Tuple
    return Tuple(CartesianIndices(nsymbols)[outer_mult_index])
end

# --------------- Build trees  ---------------
# zero leg: need S to get sector type information
function SectorFusionTree{S}() where {S <: SectorRange}
    return SectorFusionTree((), (), trivial(S), (), ())
end
function SectorFusionTree{S}(::Tuple{}, ::Tuple{}) where {S <: SectorRange}
    return SectorFusionTree((), (), trivial(S), (), ())
end

# one leg
function SectorFusionTree(sect::SectorRange, arrow::Bool)
    return SectorFusionTree((sect,), (arrow,), sect, (), ())
end

function braid_tuples(t1::Tuple{Vararg{Any, N}}, t2::Tuple{Vararg{Any, N}}) where {N}
    t12 = (t1, t2)
    nested = ntuple(i -> getindex.(t12, i), N)
    return flatten_tuples(nested)
end

function append_tree_leave(
        parent_tree::SectorFusionTree,
        branch_sector::SectorRange,
        level_arrow::Bool,
        child_root_sector,
        outer_mult,
    )
    child_leaves = (leaves(parent_tree)..., branch_sector)
    child_arrows = (arrows(parent_tree)..., level_arrow)
    child_branch_sectors = (branch_sectors(parent_tree)..., root_sector(parent_tree))
    child_outer_mul = (outer_multiplicity_indices(parent_tree)..., outer_mult)
    return SectorFusionTree(
        child_leaves, child_arrows, child_root_sector, child_branch_sectors, child_outer_mul
    )
end

function fuse_next_sector(
        parent_tree::SectorFusionTree, branch_sector::SectorRange, level_arrow::Bool
    )
    new_space = to_gradedrange(root_sector(parent_tree) ⊗ branch_sector)
    return mapreduce(
        vcat, zip(sectors(new_space), sector_multiplicities(new_space))
    ) do (la, n)
        return [
            append_tree_leave(parent_tree, branch_sector, level_arrow, la, outer_mult) for
                outer_mult in 1:n
        ]
    end
end

function recursive_build_trees(
        old_trees::Vector, sectors_to_fuse::Tuple, arrows_to_fuse::Tuple
    )
    next_level_trees = mapreduce(vcat, old_trees) do tree
        return fuse_next_sector(tree, first(sectors_to_fuse), first(arrows_to_fuse))
    end
    return recursive_build_trees(
        next_level_trees, Base.tail(sectors_to_fuse), Base.tail(arrows_to_fuse)
    )
end

function recursive_build_trees(trees::Vector, ::Tuple{}, ::Tuple{})
    return trees
end

# --------------- convert to Array  ---------------
to_array(::SectorFusionTree{<:Any, 0}) = ones(1)

function to_array(f::SectorFusionTree)
    # init with dummy trivial leg to get arrow correct and deal with size-1 case
    cgt1 = clebsch_gordan_tensor(
        trivial(sector_type(f)), first(leaves(f)), first(leaves(f)), false, first(arrows(f)), 1
    )
    tree_tensor = cgt1[1, :, :]
    return grow_tensor_tree(tree_tensor, f)
end

function to_array(f::SectorFusionTree{<:SectorProductRange})
    args = convert.(Array, arguments(f))
    return reduce(_tensor_kron, args)
end

# LinearAlgebra.kron does not allow input for ndims>2
function _tensor_kron(a::AbstractArray{<:Any, N}, b::AbstractArray{<:Any, N}) where {N}
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
function grow_tensor_tree(tree_tensor::AbstractArray{<:Real, 2}, ::SectorFusionTree{<:Any, 1})
    return tree_tensor
end

function grow_tensor_tree(
        tree_tensor::AbstractArray{<:Real, N}, f::SectorFusionTree
    ) where {N}
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
        tree_tensor::AbstractArray{<:Real, N}, f::SectorFusionTree{<:Any, N}
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
