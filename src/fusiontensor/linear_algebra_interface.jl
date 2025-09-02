# This file defines linalg for FusionTensor

using LinearAlgebra: LinearAlgebra, mul!, norm, tr

using BlockArrays: Block, blocks

using BlockSparseArrays: eachblockstoredindex
using GradedArrays: checkspaces, checkspaces_dual, quantum_dimension, sectors

# allow to contract with different eltype and let BlockSparseArray ensure compatibility
# impose matching type and number of axes at compile time
# impose matching axes at run time
function LinearAlgebra.mul!(
  C::FusionMatrix, A::FusionMatrix, B::FusionMatrix, α::Number, β::Number
)

  # compile time checks
  if ndims_domain(A) != ndims_codomain(B)
    throw(codomainError("Incompatible tensor structures for A and B"))
  end
  if ndims_codomain(A) != ndims_codomain(C)
    throw(codomainError("Incompatible tensor structures for A and C"))
  end
  if ndims_domain(B) != ndims_domain(C)
    throw(codomainError("Incompatible tensor structures for B and C"))
  end

  # input validation
  checkspaces_dual(domain_axes(A), codomain_axes(B))
  checkspaces(codomain_axes(C), codomain_axes(A))
  checkspaces(domain_axes(C), domain_axes(B))
  mul!(data_matrix(C), data_matrix(A), data_matrix(B), α, β)
  return C
end

function LinearAlgebra.norm(ft::FusionTensor)
  m = data_matrix(ft)
  row_sectors = sectors(codomain_axis(ft))
  n2 = sum(eachblockstoredindex(m); init=zero(real(eltype(ft)))) do b
    return quantum_dimension(row_sectors[Int(first(Tuple(b)))]) * norm(m[b])^2
  end
  return sqrt(n2)
end

function LinearAlgebra.tr(ft::FusionTensor)
  m = data_matrix(ft)
  row_sectors = sectors(codomain_axis(ft))
  return sum(eachblockstoredindex(m); init=zero(eltype(ft))) do b
    return quantum_dimension(row_sectors[Int(first(Tuple(b)))]) * tr(m[b])
  end
end

function LinearAlgebra.qr(ft::FusionTensor)
  qmat, rmat = block_qr(data_matrix(ft))
  qtens = FusionTensor(qmat, codomain_axes(ft), (axes(qmat, 2),))
  rtens = FusionTensor(rmat, (axes(rmat, 1),), domain_axes(ft))
  return qtens, rtens
end

function LinearAlgebra.svd(ft::FusionTensor)
  umat, s, vmat = block_svd(data_matrix(ft))
  utens = FusionTensor(umat, codomain_axes(ft), (axes(umat, 2),))
  stens = FusionTensor(s, (axes(umat, 1),), (axes(vmat, 2),))
  vtens = FusionTensor(vmat, (axes(vmat, 1),), domain_axes(ft))
  return utens, stens, vtens
end
