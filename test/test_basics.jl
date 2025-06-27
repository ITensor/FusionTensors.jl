using Test: @test, @test_throws, @testset

using BlockArrays: Block
using BlockSparseArrays: BlockSparseArray, eachblockstoredindex
using FusionTensors:
  FusionTensor,
  FusionTensorAxes,
  codomain_axes,
  data_matrix,
  domain_axes,
  FusionTensor,
  checkaxes,
  checkaxes_dual,
  codomain_axis,
  domain_axis,
  ndims_domain,
  ndims_codomain
using GradedArrays:
  U1,
  SU2,
  SectorProduct,
  TrivialSector,
  Z,
  dual,
  flip,
  gradedrange,
  sector_type,
  space_isequal
using TensorAlgebra: tuplemortar
using TensorProducts: tensor_product
using LinearAlgebra: LinearAlgebra
using Random: Random

include("setup.jl")

@testset "Fusion matrix" begin
  g1 = gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])
  g2 = dual(gradedrange([U1(0) => 2, U1(1) => 2, U1(3) => 1]))

  fta = FusionTensorAxes((g1,), (g2,))
  ft0 = FusionTensor{Float64}(undef, fta)
  @test ft0 isa FusionTensor
  @test space_isequal(codomain_axis(ft0), g1)
  @test space_isequal(domain_axis(ft0), g2)

  # check dual convention when initializing data_matrix
  ft0 = FusionTensor{Float64}(undef, (g1,), (g2,))
  @test ft0 isa FusionTensor
  @test space_isequal(codomain_axis(ft0), g1)
  @test space_isequal(domain_axis(ft0), g2)

  m = BlockSparseArray{Float64}(undef, g1, g2)
  ft1 = FusionTensor(m, (g1,), (g2,))

  # getters
  @test data_matrix(ft1) == m
  @test checkaxes(axes(ft1), tuplemortar(((g1,), (g2,))))

  # misc
  @test checkaxes(codomain_axes(ft1), (g1,))
  @test checkaxes(domain_axes(ft1), (g2,))
  @test ndims_codomain(ft1) == 1
  @test ndims_domain(ft1) == 1
  @test size(data_matrix(ft1)) == (6, 5)
  @test space_isequal(codomain_axis(ft1), g1)
  @test space_isequal(domain_axis(ft1), g2)
  @test isnothing(check_sanity(ft0))
  @test isnothing(check_sanity(ft1))
  @test sector_type(ft1) === U1{Int}
  @test sector_type(typeof(ft1)) === U1{Int}

  m1 = BlockSparseArray{Float64}(undef, g1, g2)
  m1[Block(2, 1)] = ones(2, 2)  # forbidden
  @test_throws ArgumentError FusionTensor(m1, (g1,), (g2,))

  # Base methods
  @test eltype(ft1) === Float64
  @test length(ft1) == 30
  @test ndims(ft1) == 2
  @test size(ft1) == tuplemortar(((6,), (5,)))
  @test_throws MethodError eachindex(ft1)

  # copy
  ft2 = copy(ft1)
  @test isnothing(check_sanity(ft2))
  @test ft2 !== ft1
  @test data_matrix(ft2) == data_matrix(ft1)
  @test data_matrix(ft2) !== data_matrix(ft1)
  @test checkaxes(codomain_axes(ft2), codomain_axes(ft1))
  @test checkaxes(domain_axes(ft2), domain_axes(ft1))

  ft2 = deepcopy(ft1)
  @test ft2 !== ft1
  @test data_matrix(ft2) == data_matrix(ft1)
  @test data_matrix(ft2) !== data_matrix(ft1)
  @test checkaxes(codomain_axes(ft2), codomain_axes(ft1))
  @test checkaxes(domain_axes(ft2), domain_axes(ft1))

  # similar
  ft2 = similar(ft1)
  @test isnothing(check_sanity(ft2))
  @test eltype(ft2) == Float64
  @test checkaxes(codomain_axes(ft2), codomain_axes(ft1))
  @test checkaxes(domain_axes(ft2), domain_axes(ft1))

  ft3 = similar(ft1, ComplexF64)
  @test isnothing(check_sanity(ft3))
  @test eltype(ft3) == ComplexF64
  @test checkaxes(codomain_axes(ft3), codomain_axes(ft1))
  @test checkaxes(domain_axes(ft3), domain_axes(ft1))

  @test_throws AssertionError similar(ft1, Int)

  ft5 = similar(ft1, ComplexF32, ((g1, g1), (g2,)))
  @test isnothing(check_sanity(ft5))
  @test eltype(ft5) == ComplexF64
  @test checkaxes(codomain_axes(ft5), (g1, g1))
  @test checkaxes(domain_axes(ft5), (g2,))

  ft5 = similar(ft1, ComplexF32, tuplemortar(((g1, g1), (g2,))))
  @test isnothing(check_sanity(ft5))
  @test eltype(ft5) == ComplexF64
  @test checkaxes(codomain_axes(ft5), (g1, g1))
  @test checkaxes(domain_axes(ft5), (g2,))
end

@testset "More than 2 axes" begin
  g1 = gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])
  g2 = gradedrange([U1(0) => 2, U1(1) => 2, U1(3) => 1])
  g3 = dual(gradedrange([U1(-1) => 1, U1(0) => 2, U1(1) => 1]))
  g4 = dual(gradedrange([U1(-1) => 1, U1(0) => 1, U1(1) => 1]))
  gr = tensor_product(g1, g2)
  gc = dual(tensor_product(dual(g3), dual(g4)))
  m2 = BlockSparseArray{Float64}(undef, gr, gc)
  ft = FusionTensor(m2, (g1, g2), (g3, g4))

  @test data_matrix(ft) == m2
  @test checkaxes(codomain_axes(ft), (g1, g2))
  @test checkaxes(domain_axes(ft), (g3, g4))

  @test axes(ft) == FusionTensorAxes(tuplemortar(((g1, g2), (g3, g4))))
  @test ndims_codomain(ft) == 2
  @test ndims_domain(ft) == 2
  @test size(data_matrix(ft)) == (30, 12)
  @test space_isequal(codomain_axis(ft), gr)
  @test space_isequal(domain_axis(ft), gc)
  @test isnothing(check_sanity(ft))

  @test ndims(ft) == 4
  @test size(ft) == tuplemortar(((6, 5), (4, 3)))
end

@testset "Less than 2 axes" begin
  g1 = gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])

  # one row axis
  ft1 = FusionTensor{Float64}(undef, (g1,), ())
  @test ndims_codomain(ft1) == 1
  @test ndims_domain(ft1) == 0
  @test ndims(ft1) == 1
  @test size(ft1) == tuplemortar(((6,), ()))
  @test size(data_matrix(ft1)) == (6, 1)
  @test isnothing(check_sanity(ft1))
  @test sector_type(ft1) === sector_type(g1)

  # one column axis
  ft2 = FusionTensor{Float64}(undef, (), (g1,))
  @test ndims_codomain(ft2) == 0
  @test ndims_domain(ft2) == 1
  @test ndims(ft2) == 1
  @test size(ft2) == tuplemortar(((), (6,)))
  @test size(data_matrix(ft2)) == (1, 6)
  @test isnothing(check_sanity(ft2))
  @test sector_type(ft2) === sector_type(g1)

  # zero axis
  ft3 = FusionTensor{Float64}(undef, (), ())
  @test ndims_codomain(ft3) == 0
  @test ndims_domain(ft3) == 0
  @test ndims(ft3) == 0
  @test size(ft3) == tuplemortar(((), ()))
  @test size(data_matrix(ft3)) == (1, 1)
  @test isnothing(check_sanity(ft3))
  @test sector_type(ft3) === TrivialSector
end

@testset "Base operations" begin
  g1 = gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])
  g2 = gradedrange([U1(0) => 2, U1(1) => 2, U1(3) => 1])
  g3 = gradedrange([U1(-1) => 1, U1(0) => 2, U1(1) => 1])
  g4 = gradedrange([U1(-1) => 1, U1(0) => 1, U1(1) => 1])
  ft3 = FusionTensor{Float64}(undef, (g1, g2), (g3, g4))
  @test isnothing(check_sanity(ft3))

  ft4 = +ft3
  @test ft4 === ft3  # same object

  ft4 = -ft3
  @test isnothing(check_sanity(ft4))
  @test codomain_axes(ft4) === codomain_axes(ft3)
  @test domain_axes(ft4) === domain_axes(ft3)

  ft4 = ft3 + ft3
  @test codomain_axes(ft4) === codomain_axes(ft3)
  @test domain_axes(ft4) === domain_axes(ft3)
  @test space_isequal(codomain_axis(ft4), codomain_axis(ft3))
  @test space_isequal(domain_axis(ft4), domain_axis(ft3))
  @test isnothing(check_sanity(ft4))

  ft4 = ft3 - ft3
  @test codomain_axes(ft4) === codomain_axes(ft3)
  @test domain_axes(ft4) === domain_axes(ft3)
  @test space_isequal(codomain_axis(ft4), codomain_axis(ft3))
  @test space_isequal(domain_axis(ft4), domain_axis(ft3))
  @test isnothing(check_sanity(ft4))

  ft4 = 2 * ft3
  @test codomain_axes(ft4) === codomain_axes(ft3)
  @test domain_axes(ft4) === domain_axes(ft3)
  @test space_isequal(codomain_axis(ft4), codomain_axis(ft3))
  @test space_isequal(domain_axis(ft4), domain_axis(ft3))
  @test isnothing(check_sanity(ft4))
  @test eltype(ft4) == Float64

  ft4 = 2.0 * ft3
  @test codomain_axes(ft4) === codomain_axes(ft3)
  @test domain_axes(ft4) === domain_axes(ft3)
  @test space_isequal(codomain_axis(ft4), codomain_axis(ft3))
  @test space_isequal(domain_axis(ft4), domain_axis(ft3))
  @test isnothing(check_sanity(ft4))
  @test eltype(ft4) == Float64

  ft4 = ft3 / 2.0
  @test codomain_axes(ft4) === codomain_axes(ft3)
  @test domain_axes(ft4) === domain_axes(ft3)
  @test space_isequal(codomain_axis(ft4), codomain_axis(ft3))
  @test space_isequal(domain_axis(ft4), domain_axis(ft3))
  @test isnothing(check_sanity(ft4))
  @test eltype(ft4) == Float64

  ft5 = 2.0im * ft3
  @test codomain_axes(ft5) === codomain_axes(ft3)
  @test domain_axes(ft5) === domain_axes(ft3)
  @test space_isequal(codomain_axis(ft5), codomain_axis(ft3))
  @test space_isequal(domain_axis(ft5), domain_axis(ft3))
  @test isnothing(check_sanity(ft4))
  @test eltype(ft5) == ComplexF64

  ft4 = conj(ft3)
  @test ft4 === ft3  # same object

  ft6 = conj(ft5)
  @test ft6 !== ft5  # different object
  @test isnothing(check_sanity(ft6))
  @test codomain_axes(ft6) === codomain_axes(ft5)
  @test domain_axes(ft6) === domain_axes(ft5)
  @test space_isequal(codomain_axis(ft6), codomain_axis(ft5))
  @test space_isequal(domain_axis(ft6), domain_axis(ft5))
  @test eltype(ft6) == ComplexF64

  ad = adjoint(ft3)
  @test ad isa FusionTensor
  @test ndims_codomain(ad) == 2
  @test ndims_domain(ad) == 2
  @test space_isequal(dual(g1), domain_axes(ad)[1])
  @test space_isequal(dual(g2), domain_axes(ad)[2])
  @test space_isequal(dual(g3), codomain_axes(ad)[1])
  @test space_isequal(dual(g4), codomain_axes(ad)[2])
  @test isnothing(check_sanity(ad))

  ft7 = FusionTensor{Float64}(undef, (g1,), (g2, g3, g4))
  @test_throws DimensionMismatch ft7 + ft3
  @test_throws DimensionMismatch ft7 - ft3
  @test_throws DimensionMismatch ft7 * ft3
end

@testset "specific constructors" begin
  g1 = gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])
  g2 = gradedrange([U1(0) => 2, U1(1) => 2, U1(3) => 1])
  g3 = gradedrange([U1(-1) => 1, U1(0) => 2, U1(1) => 1])
  g4 = gradedrange([U1(-1) => 1, U1(0) => 1, U1(1) => 1])

  fta = FusionTensorAxes((g1,), (g2, g3))
  @test zeros(fta) isa FusionTensor{Float64,3}
  @test zeros(ComplexF64, fta) isa FusionTensor{ComplexF64,3}

  rng = Random.default_rng()
  ft1 = randn(rng, ComplexF64, fta)
  @test ft1 isa FusionTensor{ComplexF64,3}
  @test all(!=(0), data_matrix(ft1)[Block(1, 5)])
  @test randn(rng, fta) isa FusionTensor{Float64,3}
  @test randn(ComplexF64, fta) isa FusionTensor{ComplexF64,3}
  @test randn(fta) isa FusionTensor{Float64,3}

  ft2 = FusionTensor(LinearAlgebra.I, (g1, g2))
  @test ft2 isa FusionTensor{Float64,4}
  @test axes(ft2) == FusionTensorAxes((g1, g2), dual.((g1, g2)))
  @test collect(eachblockstoredindex(data_matrix(ft2))) == map(i -> Block(i, i), 1:6)
  for i in 1:6
    m = data_matrix(ft2)[Block(i, i)]
    @test m == LinearAlgebra.I(size(m, 1))
  end

  ft2 = FusionTensor(3 * LinearAlgebra.I, (g1, g2))
  @test ft2 isa FusionTensor{Float64,4}
  @test axes(ft2) == FusionTensorAxes((g1, g2), dual.((g1, g2)))
  @test collect(eachblockstoredindex(data_matrix(ft2))) == map(i -> Block(i, i), 1:6)
  for i in 1:6
    m = data_matrix(ft2)[Block(i, i)]
    @test m == 3 * LinearAlgebra.I(size(m, 1))
  end

  @test FusionTensor{ComplexF64}(LinearAlgebra.I, (g1, g2)) isa FusionTensor{ComplexF64,4}
end

@testset "missing SectorProduct" begin
  g1 = gradedrange([SectorProduct(U1(1)) => 1])
  g2 = gradedrange([SectorProduct(U1(1), SU2(1//2)) => 1])
  g3 = gradedrange([SectorProduct(U1(1), SU2(1//2), Z{2}(1)) => 1])
  S = sector_type(g3)

  ft = FusionTensor{Float64}(undef, (g1,), (dual(g2), dual(g3)))
  @test sector_type(ft) === S
  gr = gradedrange([SectorProduct(U1(1), SU2(0), Z{2}(0)) => 1])
  @test space_isequal(codomain_axis(ft), gr)
  gc = gradedrange([
    SectorProduct(U1(2), SU2(0), Z{2}(1)) => 1, SectorProduct(U1(2), SU2(1), Z{2}(1)) => 1
  ])
  @test space_isequal(domain_axis(ft), dual(gc))

  gA = gradedrange([SectorProduct(; A=U1(1)) => 1])
  gB = gradedrange([SectorProduct(; B=SU2(1//2)) => 1])
  gC = gradedrange([SectorProduct(; C=Z{2}(0)) => 1])
  gABC = tensor_product(gA, gB, gC)
  S = sector_type(gABC)

  ft = FusionTensor{Float64}(undef, (gA, gB), (dual(gA), dual(gB), gC))
  @test sector_type(ft) === S
  @test space_isequal(codomain_axis(ft), gABC)
  @test space_isequal(domain_axis(ft), dual(gABC))
end
