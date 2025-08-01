using LinearAlgebra: norm, tr
using Test: @test, @testset

using BlockArrays: BlockArrays

using BlockSparseArrays: BlockSparseArrays
using FusionTensors: FusionTensor, to_fusiontensor
using GradedArrays: SU2, TrivialSector, U1, dual, gradedrange

include("setup.jl")

@testset "LinearAlgebra interface" begin
  sds22 = [
    0.25 0.0 0.0 0.0
    0.0 -0.25 0.5 0.0
    0.0 0.5 -0.25 0.0
    0.0 0.0 0.0 0.25
  ]
  sdst = reshape(sds22, (2, 2, 2, 2))

  g0 = gradedrange([TrivialSector() => 2])
  gu1 = gradedrange([U1(1) => 1, U1(-1) => 1])
  gsu2 = gradedrange([SU2(1 / 2) => 1])

  for g in [g0, gu1, gsu2]
    ft = to_fusiontensor(sdst, (g, g), (dual(g), dual(g)))
    @test isnothing(check_sanity(ft))
    @test norm(ft) ≈ √3 / 2
    @test isapprox(tr(ft), 0; atol=eps(Float64))
  end
end
