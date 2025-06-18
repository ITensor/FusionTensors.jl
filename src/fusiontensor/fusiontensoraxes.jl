using BlockArrays: BlockArrays
using GradedArrays:
  GradedArrays,
  AbstractGradedUnitRange,
  AbstractSector,
  TrivialSector,
  dual,
  sector_type,
  trivial
using TensorAlgebra: AbstractBlockPermutation, BlockedTuple
using TensorProducts: ⊗
using TypeParameterAccessors: type_parameters

# =======================================  Misc  ===========================================

dummy_axis() = dummy_axis(TrivialSector)
dummy_axis(x) = dummy_axis(typeof(x))
dummy_axis(::Type{T}) where {T} = dummy_axis(sector_type(T))
dummy_axis(::Type{S}) where {S<:AbstractSector} = gradedrange([trivial(S) => 1])

promote_sector_type(legs::Tuple) = promote_sector_type(legs...)

function promote_sector_type(legs...)
  # fuse trivial sectors to produce unified type
  # avoid depending on GradedArrays internals
  return sector_type(⊗(trivial.(legs)...))
end

promote_sectors(legs::NTuple{<:Any,<:AbstractGradedUnitRange}) = legs # nothing to do

function promote_sectors(legs)
  T = promote_sector_type(legs)
  # fuse with trivial to insert all missing arguments inside each GradedAxis
  # avoid depending on GradedArrays internals
  s0 = trivial(T)
  return map_sectors.(s -> only(sectors(to_gradedrange(tensor_product(s0, s)))), legs)
end

function promote_sectors(bt::BlockedTuple)
  return BlockedTuple{blocklength(bt),blocklengths(bt)}(promote_sectors(Tuple(bt)))
end

# ====================================  Definitions  =======================================

# TBD explicit axis type as type parameters?
struct FusionTensorAxes{BT<:BlockedTuple{2}}
  outer_axes::BT

  function FusionTensorAxes{BT}(bt) where {BT}
    @assert BT === typeof(promote_sectors(bt))
    return new{BT}(bt)
  end
end

# =====================================  Accessors  ========================================

outer_axes(fta::FusionTensorAxes) = fta.outer_axes

# ====================================  Constructors  ======================================

function FusionTensorAxes(bt::BlockedTuple{2})
  promoted = promote_sectors(bt)
  return FusionTensorAxes{typeof(promoted)}(promoted)
end

function FusionTensorAxes(codomain_legs, domain_legs)
  return FusionTensorAxes(tuplemortar((codomain_legs, domain_legs)))
end

# ==================================  Base interface  ======================================

for f in [
  :(broadcastable), :(Tuple), :(axes), :(firstindex), :(lastindex), :(iterate), :(length)
]
  @eval Base.$f(fta::FusionTensorAxes) = Base.$f(outer_axes(fta))
end

for f in [:(getindex), :(iterate)]
  @eval Base.$f(fta::FusionTensorAxes, i) = $f(outer_axes(fta), i)
end

function Base.getindex(fta::FusionTensorAxes, bp::AbstractBlockPermutation)
  return FusionTensorAxes(outer_axes(fta)[bp])
end

Base.copy(fta::FusionTensorAxes) = FusionTensorAxes(copy.(outer_axes(fta)))

Base.deepcopy(fta::FusionTensorAxes) = FusionTensorAxes(deepcopy.(outer_axes(fta)))

function Base.:(==)(a::FusionTensorAxes, b::FusionTensorAxes)
  blocklengths(a) != blocklengths(b) && return false
  for i in 1:length(a)
    !space_isequal(a[i], b[i]) && return false
  end
  return true
end

# ================================  BlockArrays interface  =================================

for f in [:(blocklength), :(blocklengths), :(blocks)]
  @eval BlockArrays.$f(fta::FusionTensorAxes) = $f(outer_axes(fta))
end

# ==============================  GradedArrays interface  ==================================

function GradedArrays.sector_type(
  ::Type{FTA}
) where {BT<:BlockedTuple{2,(0, 0)},FTA<:FusionTensorAxes{BT}}
  return TrivialSector
end

function GradedArrays.sector_type(::Type{FTA}) where {BT,FTA<:FusionTensorAxes{BT}}
  return sector_type(type_parameters(type_parameters(BT, 3), 1))
end

# ==============================  FusionTensor interface  ==================================

codomain_axes(fta::FusionTensorAxes) = fta[Block(1)]

domain_axes(fta::FusionTensorAxes) = fta[Block(2)]

function codomain_axis(fta::FusionTensorAxes)
  if ndims_codomain(fta) == 0
    return dummy_axis(fta)
  end
  return ⊗(codomain_axes(fta)...)
end

function domain_axis(fta::FusionTensorAxes)
  if ndims_domain(fta) == 0
    return dual(dummy_axis(fta))
  end
  return dual(⊗(dual.(domain_axes(fta))...))
end

ndims_codomain(fta::FusionTensorAxes) = length(codomain_axes(fta))

ndims_domain(fta::FusionTensorAxes) = length(domain_axes(fta))
