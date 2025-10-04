using LinearAlgebra: norm, normalize, normalize!, tr
using Test: @test, @testset

using BlockArrays: BlockArrays

using BlockSparseArrays: BlockSparseArrays
using FusionTensors: FusionTensor, to_fusiontensor
using GradedArrays: O2, SU2, TrivialSector, U1, dual, gradedrange

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
    go2 = gradedrange([O2(1 / 2) => 1])
    gsu2 = gradedrange([SU2(1 / 2) => 1])

    for g in [g0, gu1, go2, gsu2]
        ft = to_fusiontensor(sdst, (g, g), (dual(g), dual(g)))
        @test isnothing(check_sanity(ft))
        @test norm(ft) ≈ √3 / 2
        @test norm(ft, 2) ≈ √3 / 2
        @test norm(ft, 2.0) ≈ √3 / 2
        @test isapprox(tr(ft), 0; atol = eps(Float64))

        ft2 = normalize(ft)
        @test norm(ft2) ≈ 1.0
        @test norm(ft) ≈ √3 / 2  # unaffected by normalize
        @test ft ≈ √3 / 2 * ft2
        normalize!(ft)
        @test norm(ft) ≈ 1.0
    end

    for g in [g0, gu1]
        ft = to_fusiontensor(sdst, (g, g), (dual(g), dual(g)))
        @test norm(ft, 1) ≈ 2.0
    end
    for g in [go2, gsu2]
        ft = to_fusiontensor(sdst, (g, g), (dual(g), dual(g)))
        @test norm(ft, 1) ≈ 1.5
    end
end
