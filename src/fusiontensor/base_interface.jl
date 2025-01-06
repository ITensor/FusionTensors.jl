# This files defines Base functions for FusionTensor

using Accessors: @set

using BlockSparseArrays: @view!

set_data_matrix(ft::FusionTensor, data_matrix) = @set ft.data_matrix = data_matrix

Base.:*(x::Number, ft::FusionTensor) = set_data_matrix(ft, x * data_matrix(ft))
Base.:*(ft::FusionTensor, x::Number) = set_data_matrix(ft, x * data_matrix(ft))

# tensor contraction is a block data_matrix product.
function Base.:*(left::FusionTensor, right::FusionTensor)
  @assert matching_dual(domain_axes(left), codomain_axes(right))
  new_data_matrix = data_matrix(left) * data_matrix(right)
  return fusiontensor(new_data_matrix, codomain_axes(left), domain_axes(right))
end

Base.:+(ft::FusionTensor) = ft

# tensor addition is a block data_matrix add.
function Base.:+(left::FusionTensor, right::FusionTensor)
  @assert matching_axes(axes(left), axes(right))
  return set_data_matrix(left, data_matrix(left) + data_matrix(right))
end

Base.:-(ft::FusionTensor) = set_data_matrix(ft, -data_matrix(ft))

function Base.:-(left::FusionTensor, right::FusionTensor)
  @assert matching_axes(axes(left), axes(right))
  return set_data_matrix(left, data_matrix(left) - data_matrix(right))
end

Base.:/(ft::FusionTensor, x::Number) = set_data_matrix(ft, data_matrix(ft) / x)

Base.Array(ft::FusionTensor) = Array(to_array(ft))

# adjoint is costless: dual axes, swap codomain and domain, take data_matrix adjoint.
# data_matrix coeff are not modified (beyond complex conjugation)
transpose_mapping(d::Dict) = Dict([reverse(k) => transpose_mapping(v) for (k, v) in d])
function transpose_mapping(b::BlockIndexRange{2})
  new_block = Block(reverse(Tuple(Block(b))))
  return new_block[reverse(b.indices)...]
end
function Base.adjoint(ft::FusionTensor)
  return FusionTensor(
    adjoint(data_matrix(ft)),
    dual.(domain_axes(ft)),
    dual.(codomain_axes(ft)),
    transpose_mapping(trees_block_mapping(ft)),
  )
end

Base.axes(ft::FusionTensor) = (codomain_axes(ft)..., domain_axes(ft)...)

# conj is defined as coefficient wise complex conjugation, without axis dual
Base.conj(ft::FusionTensor{<:Real}) = ft   # same object for real element type
Base.conj(ft::FusionTensor) = set_data_matrix(ft, conj(data_matrix(ft)))

function Base.copy(ft::FusionTensor)
  return FusionTensor(
    copy(data_matrix(ft)),
    copy.(codomain_axes(ft)),
    copy.(domain_axes(ft)),
    copy(trees_block_mapping(ft)),
  )
end

function Base.deepcopy(ft::FusionTensor)
  return FusionTensor(
    deepcopy(data_matrix(ft)),
    deepcopy.(codomain_axes(ft)),
    deepcopy.(domain_axes(ft)),
    deepcopy(trees_block_mapping(ft)),
  )
end

# eachindex is automatically defined for AbstractArray. We do not want it.
Base.eachindex(::FusionTensor) = error("eachindex not defined for FusionTensor")

function Base.getindex(ft::FusionTensor, f1::SectorFusionTree, f2::SectorFusionTree)
  charge_matrix = data_matrix(ft)[trees_block_mapping(ft)[f1, f2]]
  return reshape(charge_matrix, charge_block_size(ft, f1, f2))
end

function Base.setindex!(
  ft::FusionTensor, a::AbstractArray, f1::SectorFusionTree, f2::SectorFusionTree
)
  return view(ft, f1, f2) .= a
end

Base.permutedims(ft::FusionTensor, args...) = fusiontensor_permutedims(ft, args...)

Base.similar(ft::FusionTensor) = similar(ft, eltype(ft))
function Base.similar(ft::FusionTensor, ::Type{T}) where {T}
  # fusion trees have Float64 eltype: need compatible type
  @assert promote_type(T, Float64) === T
  mat = similar(data_matrix(ft), T)
  initialize_allowed_sectors!(mat)
  return FusionTensor(mat, codomain_axes(ft), domain_axes(ft), trees_block_mapping(ft))
end

function Base.similar(::FusionTensor, ::Type{T}, new_axes::Tuple{<:Tuple,<:Tuple}) where {T}
  return FusionTensor(T, new_axes[1], new_axes[2])
end

Base.show(io::IO, ft::FusionTensor) = print(io, "$(ndims(ft))-dim FusionTensor")

function Base.show(io::IO, ::MIME"text/plain", ft::FusionTensor)
  println(io, "$(ndims(ft))-dim FusionTensor with axes:")
  for ax in axes(ft)
    println(io, ax)
  end
  return nothing
end

Base.size(ft::FusionTensor) = quantum_dimension.(axes(ft))

function Base.view(ft::FusionTensor, f1::SectorFusionTree, f2::SectorFusionTree)
  charge_matrix = @view! data_matrix(ft)[trees_block_mapping(ft)[f1, f2]]
  return reshape(charge_matrix, charge_block_size(ft, f1, f2))
end
