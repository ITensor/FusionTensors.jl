using LinearAlgebra: Adjoint

using BlockSparseArrays: BlockSparseMatrix, eachblockstoredindex
using FusionTensors:
  FusionTensor,
  codomain_axes,
  data_matrix,
  domain_axes,
  checkaxes,
  checkaxes_dual,
  matrix_column_axis,
  matrix_row_axis,
  ndims_codomain,
  ndims_domain
using GradedUnitRanges: blocklabels, dual, space_isequal

function check_data_matrix_axes(
  mat::BlockSparseMatrix, codomain_legs::Tuple, domain_legs::Tuple
)
  ft0 = FusionTensor(Float64, codomain_legs, domain_legs)
  @assert space_isequal(matrix_row_axis(ft0), axes(mat, 1))
  @assert space_isequal(matrix_column_axis(ft0), axes(mat, 2))
end

function check_data_matrix_axes(mat::Adjoint, codomain_legs::Tuple, domain_legs::Tuple)
  return check_data_matrix_axes(adjoint(mat), dual.(domain_legs), dual.(codomain_legs))
end

function check_sanity(ft::FusionTensor)
  nca = ndims_domain(ft)
  @assert nca == length(domain_axes(ft)) "ndims_domain does not match domain_axes"
  @assert nca <= ndims(ft) "invalid ndims_domain"

  nda = ndims_codomain(ft)
  @assert nda == length(codomain_axes(ft)) "ndims_codomain does not match codomain_axes"
  @assert nda <= ndims(ft) "invalid ndims_codomain"
  @assert nda + nca == ndims(ft) "invalid ndims"

  @assert length(axes(ft)) == ndims(ft) "ndims does not match axes"
  checkaxes(axes(ft)[begin:nda], codomain_axes(ft))
  checkaxes(axes(ft)[(nda + 1):end], domain_axes(ft))

  m = data_matrix(ft)
  @assert ndims(m) == 2 "invalid data_matrix ndims"
  row_axis = matrix_row_axis(ft)
  column_axis = matrix_column_axis(ft)
  @assert row_axis === axes(m, 1) "invalid row_axis"
  @assert column_axis === axes(m, 2) "invalid column_axis"
  check_data_matrix_axes(m, codomain_axes(ft), domain_axes(ft))

  for b in eachblockstoredindex(m)
    ir, ic = Int.(Tuple(b))
    @assert blocklabels(row_axis)[ir] == blocklabels(dual(column_axis))[ic] "forbidden block"
  end
  return nothing
end
