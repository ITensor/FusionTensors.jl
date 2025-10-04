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

function LinearAlgebra.norm(ft::FusionTensor, p::Real = 2)
    m = data_matrix(ft)
    row_sectors = sectors(codomain_axis(ft))
    np = sum(eachblockstoredindex(m); init = zero(real(eltype(ft)))) do b
        return quantum_dimension(row_sectors[Int(first(Tuple(b)))]) * norm(m[b], p)^p
    end
    return np^(1 / p)
end

LinearAlgebra.normalize(ft::FusionTensor, p::Real = 2) = ft / norm(ft, p)

function LinearAlgebra.normalize!(ft::FusionTensor, p::Real = 2)
    data_matrix(ft) ./= norm(ft, p)
    return ft
end

function LinearAlgebra.tr(ft::FusionTensor)
    m = data_matrix(ft)
    row_sectors = sectors(codomain_axis(ft))
    return sum(eachblockstoredindex(m); init = zero(eltype(ft))) do b
        return quantum_dimension(row_sectors[Int(first(Tuple(b)))]) * tr(m[b])
    end
end
