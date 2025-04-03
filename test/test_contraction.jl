using LinearAlgebra: mul!
using Test: @test, @testset, @test_broken

using BlockSparseArrays: BlockSparseArray
using FusionTensors: FusionTensor, domain_axes, codomain_axes
using GradedArrays: dual, gradedrange
using GradedArrays.SymmetrySectors: U1
using TensorAlgebra: contract

include("setup.jl")

@testset "contraction" begin
  g1 = gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])
  g2 = gradedrange([U1(0) => 2, U1(1) => 2, U1(3) => 1])
  g3 = gradedrange([U1(-1) => 1, U1(0) => 2, U1(1) => 1])
  g4 = gradedrange([U1(-1) => 1, U1(0) => 1, U1(1) => 1])

  ft1 = FusionTensor(Float64, (g1, g2), (g3, g4))
  @test isnothing(check_sanity(ft1))

  ft2 = FusionTensor(Float64, dual.((g3, g4)), (g1,))
  @test isnothing(check_sanity(ft2))

  ft3 = ft1 * ft2  # tensor contraction
  @test isnothing(check_sanity(ft3))
  @test domain_axes(ft3) === domain_axes(ft2)
  @test codomain_axes(ft3) === codomain_axes(ft1)

  # test LinearAlgebra.mul! with in-place matrix product
  mul!(ft3, ft1, ft2)
  @test isnothing(check_sanity(ft3))
  @test domain_axes(ft3) === domain_axes(ft2)
  @test codomain_axes(ft3) === codomain_axes(ft1)

  mul!(ft3, ft1, ft2, 1.0, 1.0)
  @test isnothing(check_sanity(ft2))
  @test domain_axes(ft3) === domain_axes(ft2)
  @test codomain_axes(ft3) === codomain_axes(ft1)
end

@testset "TensorAlgebra interface" begin
  g1 = gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])
  g2 = gradedrange([U1(0) => 2, U1(1) => 2, U1(3) => 1])
  g3 = gradedrange([U1(-1) => 1, U1(0) => 2, U1(1) => 1])
  g4 = gradedrange([U1(-1) => 1, U1(0) => 1, U1(1) => 1])

  ft1 = FusionTensor(Float64, (g1, g2), (g3, g4))
  ft2 = FusionTensor(Float64, dual.((g3, g4)), (dual(g1),))
  ft3 = FusionTensor(Float64, dual.((g3, g4)), dual.((g1, g2)))

  ft4, legs = contract(ft1, (1, 2, 3, 4), ft2, (3, 4, 5))
  @test legs == (1, 2, 5)
  @test isnothing(check_sanity(ft4))
  @test domain_axes(ft4) === domain_axes(ft2)
  @test codomain_axes(ft4) === codomain_axes(ft1)

  ft5 = contract((1, 2, 5), ft1, (1, 2, 3, 4), ft2, (3, 4, 5))
  @test isnothing(check_sanity(ft5))

  # biperm is not allowed
  @test_broken contract(((1, 2), (5,)), ft1, (1, 2, 3, 4), ft2, (3, 4, 5))

  @test permutedims(ft1, (), (1, 2, 3, 4)) * permutedims(ft3, (3, 4, 1, 2), ()) isa
    FusionTensor{Float64,0}
  @test_broken contract(ft1, (1, 2, 3, 4), ft3, (3, 4, 1, 2))
end
