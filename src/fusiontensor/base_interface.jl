# This files defines Base functions for FusionTensor

function Base.:*(x::Number, ft::FusionTensor)
  return FusionTensor(x * data_matrix(ft), codomain_axes(ft), domain_axes(ft))
end

function Base.:*(ft::FusionTensor, x::Number)
  return FusionTensor(x * data_matrix(ft), codomain_axes(ft), domain_axes(ft))
end

# tensor contraction is a block data_matrix product.
# allow to contract with different eltype and let BlockSparseArray ensure compatibility
# impose matching type and number of axes at compile time
# impose matching axes at run time
function Base.:*(left::FusionTensor, right::FusionTensor)
  @asssert matching_dual(domain_axes(left), codomain_axes(right))
  new_data_matrix = data_matrix(left) * data_matrix(right)
  return FusionTensor(new_data_matrix, codomain_axes(left), domain_axes(right))
end

Base.:+(ft::FusionTensor) = ft

# tensor addition is a block data_matrix add.
# impose matching axes, allow different eltypes
function Base.:+(left::FusionTensor, right::FusionTensor)
  @assert !matching_axes(axes(left), axes(right))
  new_data_matrix = data_matrix(left) + data_matrix(right)
  return FusionTensor(new_data_matrix, codomain_axes(left), domain_axes(left))
end

function Base.:-(ft::FusionTensor)
  new_data_matrix = -data_matrix(ft)
  return FusionTensor(new_data_matrix, codomain_axes(ft), domain_axes(ft))
end

function Base.:-(left::FusionTensor, right::FusionTensor)
  @assert matching_axes(axes(left), axes(right))
  new_data_matrix = data_matrix(left) - data_matrix(right)
  return FusionTensor(new_data_matrix, codomain_axes(left), domain_axes(left))
end

function Base.:/(ft::FusionTensor, x::Number)
  return FusionTensor(data_matrix(ft) / x, codomain_axes(ft), domain_axes(ft))
end

Base.Array(ft::FusionTensor) = Array(cast_to_array(ft))

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
function Base.conj(ft::FusionTensor)
  return FusionTensor(conj(data_matrix(ft)), codomain_axes(ft), domain_axes(ft))
end

function Base.copy(ft::FusionTensor)
  new_data_matrix = copy(data_matrix(ft))
  new_codomain_axes = copy.(codomain_axes(ft))
  new_domain_axes = copy.(domain_axes(ft))
  return FusionTensor(new_data_matrix, new_codomain_axes, new_domain_axes)
end

function Base.deepcopy(ft::FusionTensor)
  new_data_matrix = deepcopy(data_matrix(ft))
  new_codomain_axes = deepcopy.(codomain_axes(ft))
  new_domain_axes = deepcopy.(domain_axes(ft))
  return FusionTensor(new_data_matrix, new_codomain_axes, new_domain_axes)
end

# eachindex is automatically defined for AbstractArray. We do not want it.
Base.eachindex(::FusionTensor) = error("eachindex not defined for FusionTensor")

Base.getindex(ft::FusionTensor, f1f2::Tuple{<:FusionTree,<:FusionTree}) = ft[f1f2...]
function Base.getindex(ft::FusionTensor, f1::FusionTree, f2::FusionTree)
  return data_matrix(ft)[trees_block_mapping(ft)[f1, f2]]
end

function Base.setindex!(
  ft::FusionTensor, m::AbstractMatrix, f1f2::Tuple{<:FusionTree,<:FusionTree}
)
  return setindex!(ft, m, f1f2...)
end
# TBD any way to replace explicit definition with better handling of @views?
function Base.setindex!(ft::FusionTensor, m::AbstractMatrix, f1::FusionTree, f2::FusionTree)
  #return setindex!(data_matrix(ft), m, trees_block_mapping(ft)[f1, f2])
  # workaround for setindex(::BlockSparseArray) issue
  return data_matrix(ft)[trees_block_mapping(ft)[f1, f2]] .= m
end

Base.ndims(::FusionTensor{T,N}) where {T,N} = N

Base.permutedims(ft::FusionTensor, args...) = fusiontensor_permutedims(ft, args...)

function Base.similar(ft::FusionTensor)
  mat = similar(data_matrix(ft))
  return FusionTensor(mat, codomain_axes(ft), domain_axes(ft))
end

function Base.similar(ft::FusionTensor, elt::Type)
  return FusionTensor(elt, codomain_axes(ft), domain_axes(ft))
end

function Base.similar(::FusionTensor, elt::Type, new_axes::Tuple{<:Tuple,<:Tuple})
  return FusionTensor(elt, new_axes[1], new_axes[2])
end

Base.show(io::IO, ft::FusionTensor) = print(io, "$(ndims(ft))-dim FusionTensor")

function Base.show(io::IO, ::MIME"text/plain", ft::FusionTensor)
  println(io, "$(ndims(ft))-dim FusionTensor with axes:")
  for ax in axes(ft)
    display(ax)
    println(io)
  end
  return nothing
end

Base.size(ft::FusionTensor) = quantum_dimension.(axes(ft))

Base.view(ft::FusionTensor, f1f2::Tuple{<:FusionTree,<:FusionTree}) = view(ft, f1f2...)
function Base.view(ft::FusionTensor, f1::FusionTree, f2::FusionTree)
  return view(data_matrix(ft), trees_block_mapping(ft)[f1, f2])
end
