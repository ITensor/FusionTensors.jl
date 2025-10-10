using BlockArrays: BlockArrays
using GradedArrays:
    GradedArrays,
    AbstractGradedUnitRange,
    SymmetryStyle,
    TrivialSector,
    dual,
    sector_type,
    trivial
using TensorAlgebra:
    TensorAlgebra,
    AbstractBlockPermutation,
    AbstractBlockTuple,
    BlockedTuple,
    length_codomain,
    length_domain
using TensorProducts: ⊗
using TypeParameterAccessors: type_parameters

# =======================================  Misc  ===========================================

promote_sector_type(legs::Tuple) = promote_sector_type(legs...)

function promote_sector_type(legs...)
    # fuse trivial sectors to produce unified type
    # avoid depending on GradedArrays internals
    return sector_type(⊗(trivial.(legs)...))
end

promote_sectors(legs::NTuple{<:Any, <:AbstractGradedUnitRange}) = legs # nothing to do

function promote_sectors(legs)
    T = promote_sector_type(legs)
    # fuse with trivial to insert all missing arguments inside each GradedAxis
    # avoid depending on GradedArrays internals
    s0 = trivial(T)
    return map_sectors.(s -> only(sectors(to_gradedrange(tensor_product(s0, s)))), legs)
end

function promote_sectors(bt::BlockedTuple)
    return BlockedTuple{blocklength(bt), blocklengths(bt)}(promote_sectors(Tuple(bt)))
end

# ====================================  Definitions  =======================================

# TBD explicit axis type as type parameters?
struct FusionTensorAxes{BT <: BlockedTuple{2}}
    outer_axes::BT

    function FusionTensorAxes{BT}(bt) where {BT}
        @assert BT === typeof(promote_sectors(bt))
        return new{BT}(bt)
    end
end

# ====================================  Constructors  ======================================

function FusionTensorAxes(bt::BlockedTuple{2})
    promoted = promote_sectors(bt)
    return FusionTensorAxes{typeof(promoted)}(promoted)
end

function FusionTensorAxes(codomain_legs, domain_legs)
    return FusionTensorAxes(tuplemortar((codomain_legs, domain_legs)))
end

# ==============================  TensorAlgebra interface  =================================

TensorAlgebra.BlockedTuple(fta::FusionTensorAxes) = fta.outer_axes

TensorAlgebra.trivial_axis(fta::FusionTensorAxes) = trivial_axis(sector_type(fta))

TensorAlgebra.length_domain(fta::FusionTensorAxes) = length(domain(fta))

# ==================================  Base interface  ======================================

for f in [
        :(broadcastable), :(Tuple), :(axes), :(firstindex), :(lastindex), :(iterate), :(length),
    ]
    @eval Base.$f(fta::FusionTensorAxes) = Base.$f(BlockedTuple(fta))
end

for f in [:(getindex), :(iterate)]
    @eval Base.$f(fta::FusionTensorAxes, i) = $f(BlockedTuple(fta), i)
end

function Base.getindex(fta::FusionTensorAxes, bp::AbstractBlockPermutation)
    return FusionTensorAxes(BlockedTuple(fta)[bp])
end

Base.copy(fta::FusionTensorAxes) = FusionTensorAxes(copy.(BlockedTuple(fta)))

Base.deepcopy(fta::FusionTensorAxes) = FusionTensorAxes(deepcopy.(BlockedTuple(fta)))

function Base.:(==)(a::FusionTensorAxes, b::FusionTensorAxes)
    blocklengths(a) != blocklengths(b) && return false
    for i in 1:length(a)
        !space_isequal(a[i], b[i]) && return false
    end
    return true
end

# ================================  BlockArrays interface  =================================

for f in [:(blocklength), :(blocklengths), :(blocks)]
    @eval BlockArrays.$f(fta::FusionTensorAxes) = $f(BlockedTuple(fta))
end

# ==============================  GradedArrays interface  ==================================

function GradedArrays.sector_type(
        ::Type{FTA}
    ) where {BT <: BlockedTuple{2, (0, 0)}, FTA <: FusionTensorAxes{BT}}
    return TrivialSector
end

function GradedArrays.sector_type(::Type{FTA}) where {BT, FTA <: FusionTensorAxes{BT}}
    return sector_type(type_parameters(type_parameters(BT, 3), 1))
end

function GradedArrays.SymmetryStyle(::Type{FTA}) where {FTA <: FusionTensorAxes}
    return SymmetryStyle(sector_type(FTA))
end

function GradedArrays.checkspaces(
        ::Type{Bool}, left::FusionTensorAxes, right::FusionTensorAxes
    )
    return left == right
end

# ==============================  FusionTensor interface  ==================================

codomain(fta::FusionTensorAxes) = fta[Block(1)]

domain(fta::FusionTensorAxes) = fta[Block(2)]

function fused_codomain(fta::FusionTensorAxes)
    if length_codomain(fta) == 0
        return trivial_axis(fta)
    end
    return ⊗(codomain(fta)...)
end

function fused_domain(fta::FusionTensorAxes)
    if length_domain(fta) == 0
        return dual(trivial_axis(fta))
    end
    return dual(⊗(dual.(domain(fta))...))
end
