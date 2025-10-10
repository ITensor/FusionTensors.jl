# This file defines struct FusionTensor and constructors

using BlockArrays: AbstractBlockMatrix, BlockArrays, BlockIndexRange, blocklength, findblock

using BlockSparseArrays:
    AbstractBlockSparseMatrix, BlockSparseArray, eachblockstoredindex, to_block_indices
using GradedArrays:
    AbstractGradedUnitRange,
    SymmetryStyle,
    TrivialSector,
    dual,
    findfirstblock,
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
using Random: Random, AbstractRNG, randn!
using TensorAlgebra: BlockedTuple, trivial_axis, tuplemortar, length_codomain, length_domain
using TensorProducts: tensor_product
using TypeParameterAccessors: type_parameters

# =======================================  Misc  ===========================================
function flip_domain(nonflipped_col_axis, nonflipped_trees_to_ranges)
    col_axis = dual(nonflipped_col_axis)
    domain_trees_to_ranges_mapping = Dict(
        map(((tree, v),) -> flip(tree) => v, collect(nonflipped_trees_to_ranges))
    )
    return col_axis, domain_trees_to_ranges_mapping
end

function fuse_axes(::Type{S}, ::Tuple{}) where {S <: SectorRange}
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
    range_mapping = Dict{type_parameters(eltype(fusion_trees_mult), 1), typeof(Block(1)[1:1])}()
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
        codomain_trees_to_ranges_mapping::Dict{<:SectorFusionTree, <:BlockIndexRange{1}},
        domain_trees_to_ranges_mapping::Dict{<:SectorFusionTree, <:BlockIndexRange{1}},
    )
    return Dict(
        map(
            Iterators.filter(
                t -> root_sector(first(first(t))) == dual(root_sector(first(t[2]))),
                Iterators.product(codomain_trees_to_ranges_mapping, domain_trees_to_ranges_mapping),
            ),
        ) do t
            return first.(t) => BlockIndexRange(last.(t))
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

struct FusionTensor{T, N, Axes <: FusionTensorAxes, Mat <: AbstractMatrix{T}, Mapping} <:
    AbstractArray{T, N}
    data_matrix::Mat
    axes::Axes
    trees_block_mapping::Mapping

    # inner constructor to impose constraints on types
    function FusionTensor{T, N, Axes, Mat, Mapping}(
            mat, legs, trees_block_mapping
        ) where {T, N, Axes, Mat, Mapping}
        S = sector_type(legs)
        @assert keytype(trees_block_mapping) <:
        Tuple{<:SectorFusionTree{S}, <:SectorFusionTree{S}}
        return new{T, N, Axes, Mat, Mapping}(mat, legs, trees_block_mapping)
    end
end

const FusionMatrix{T, Axes, Mat, Mapping} = FusionTensor{
    T, 2, Axes, Mapping,
} where {BT <: BlockedTuple{2, (1, 1)}, Axes <: FusionTensorAxes{BT}}

# =====================================  Accessors  ========================================

data_matrix(ft::FusionTensor) = ft.data_matrix
trees_block_mapping(ft::FusionTensor) = ft.trees_block_mapping

# ====================================  Constructors  ======================================

function FusionTensor(
        mat::AbstractMatrix,
        legs::FusionTensorAxes,
        trees_block_mapping::Dict{<:Tuple{<:SectorFusionTree, <:SectorFusionTree}},
    )
    return FusionTensor{
        eltype(mat), length(legs), typeof(legs), typeof(mat), typeof(trees_block_mapping),
    }(
        mat, legs, trees_block_mapping
    )
end

# empty matrix
function FusionTensor{T}(::UndefInitializer, legs::FusionTensorAxes) where {T}
    S = sector_type(legs)
    row_axis, codomain_trees_to_ranges = fuse_axes(S, codomain(legs))
    col_axis, domain_trees_to_ranges = flip_domain(fuse_axes(S, dual.(domain(legs)))...)

    mat = initialize_data_matrix(T, row_axis, col_axis)
    tree_to_block_mapping = intersect_codomain_domain(
        codomain_trees_to_ranges, domain_trees_to_ranges
    )
    return FusionTensor(mat, legs, tree_to_block_mapping)
end

#constructor from precomputed data_matrix
function FusionTensor(mat::AbstractMatrix, legs::FusionTensorAxes)
    # init with empty data_matrix to construct trees_block_mapping
    ft = FusionTensor{eltype(mat)}(undef, legs)
    for b in eachblockstoredindex(mat)
        b in eachblockstoredindex(data_matrix(ft)) ||
            throw(ArgumentError("matrix block $b is not allowed"))
        data_matrix(ft)[b] = mat[b]
    end
    return ft
end

FusionTensor(x, legs::BlockedTuple{2}) = FusionTensor(x, FusionTensorAxes(legs))
function FusionTensor{T}(x, legs::BlockedTuple{2}) where {T}
    return FusionTensor{T}(x, FusionTensorAxes(legs))
end

# constructor from split axes
function FusionTensor(
        x,
        codomain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
        domain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
    )
    return FusionTensor(x, tuplemortar((codomain_legs, domain_legs)))
end

function FusionTensor{T}(
        x,
        codomain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
        domain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
    ) where {T}
    return FusionTensor{T}(x, tuplemortar((codomain_legs, domain_legs)))
end

# specific constructors
function Base.zeros(::Type{T}, fta::FusionTensorAxes) where {T}
    ft = FusionTensor{T}(undef, fta)
    foreach(m -> fill!(m, zero(T)), eachstoredblock(data_matrix(ft)))
    return ft
end
Base.zeros(fta::FusionTensorAxes) = zeros(Float64, fta)

function Base.randn(rng::AbstractRNG, ::Type{T}, fta::FusionTensorAxes) where {T}
    ft = FusionTensor{T}(undef, fta)
    foreach(m -> randn!(rng, m), eachstoredblock(data_matrix(ft)))
    return ft
end
Base.randn(rng::AbstractRNG, fta::FusionTensorAxes) = randn(rng, Float64, fta)
Base.randn(::Type{T}, fta::FusionTensorAxes) where {T} = randn(Random.default_rng(), T, fta)
Base.randn(fta::FusionTensorAxes) = randn(Float64, fta)

function FusionTensor{T}(
        s::UniformScaling, codomain_legs::Tuple{Vararg{AbstractGradedUnitRange}}
    ) where {T}
    fta = FusionTensorAxes(codomain_legs, dual.(codomain_legs))
    ft = FusionTensor{T}(undef, fta)
    for m in eachstoredblock(data_matrix(ft))
        m .= s(size(m, 1))
    end
    return ft
end
function FusionTensor(
        s::UniformScaling, codomain_legs::Tuple{Vararg{AbstractGradedUnitRange}}
    )
    return FusionTensor{Float64}(s, codomain_legs)
end

# ================================  BlockArrays interface  =================================

function BlockArrays.findblock(ft::FusionTensor, f1::SectorFusionTree, f2::SectorFusionTree)
    # find outer block corresponding to fusion trees
    @assert typeof((f1, f2)) === keytype(trees_block_mapping(ft))
    b1 = findfirstblock.(flip_dual.(codomain_axes(ft)), leaves(f1))
    b2 = findfirstblock.(flip_dual.(domain_axes(ft)), leaves(f2))
    return Block(Int.(b1)..., Int.(b2)...)
end

# ==============================  GradedArrays interface  ==================================

function GradedArrays.sector_type(::Type{FT}) where {FT <: FusionTensor}
    return sector_type(type_parameters(FT, 3))
end

function GradedArrays.SymmetryStyle(::Type{FT}) where {FT <: FusionTensor}
    return SymmetryStyle(sector_type(FT))
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
