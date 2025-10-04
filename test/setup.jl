using LinearAlgebra: Adjoint
using BlockArrays: blocklengths
using BlockSparseArrays: BlockSparseMatrix, eachblockstoredindex
using FusionTensors:
    FusionTensor,
    codomain_axes,
    data_matrix,
    domain_axes,
    domain_axis,
    codomain_axis,
    ndims_codomain,
    ndims_domain
using GradedArrays:
    checkspaces, checkspaces_dual, dual, sectors, sector_multiplicities, space_isequal

function check_sanity(ft::FusionTensor)
    nca = ndims_domain(ft)
    @assert nca == length(domain_axes(ft)) "ndims_domain does not match domain_axes"
    @assert nca <= ndims(ft) "invalid ndims_domain"

    nda = ndims_codomain(ft)
    @assert nda == length(codomain_axes(ft)) "ndims_codomain does not match codomain_axes"
    @assert nda <= ndims(ft) "invalid ndims_codomain"
    @assert nda + nca == ndims(ft) "invalid ndims"

    @assert length(axes(ft)) == ndims(ft) "ndims does not match axes"
    checkspaces(axes(ft)[begin:nda], codomain_axes(ft))
    checkspaces(axes(ft)[(nda + 1):end], domain_axes(ft))

    m = data_matrix(ft)
    @assert ndims(m) == 2 "invalid data_matrix ndims"
    row_axis = codomain_axis(ft)
    column_axis = domain_axis(ft)
    @assert sector_multiplicities(row_axis) == blocklengths(axes(m, 1)) "invalid row_axis"
    @assert sector_multiplicities(column_axis) == blocklengths(axes(m, 2)) "invalid column_axis"

    for b in eachblockstoredindex(m)
        ir, ic = Int.(Tuple(b))
        @assert sectors(row_axis)[ir] == sectors(dual(column_axis))[ic] "forbidden block"
    end
    return nothing
end
