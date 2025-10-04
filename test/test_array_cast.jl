using LinearAlgebra: LinearAlgebra, norm
using Test: @test, @test_throws, @testset

using BlockArrays: Block, BlockedArray, blocksize

using FusionTensors: FusionTensor, data_matrix, to_fusiontensor
using GradedArrays: O2, SectorProduct, SU2, TrivialSector, U1, dual, gradedrange
using TensorProducts: tensor_product

include("setup.jl")

@testset "Trivial FusionTensor" begin
    @testset "trivial matrix" begin
        g = gradedrange([TrivialSector() => 1])
        gb = dual(g)
        m = ones((1, 1))
        ft = to_fusiontensor(m, (g,), (gb,))
        @test size(data_matrix(ft)) == (1, 1)
        @test blocksize(data_matrix(ft)) == (1, 1)
        @test data_matrix(ft)[1, 1] ≈ 1.0
        @test isnothing(check_sanity(ft))
        @test Array(ft) ≈ m
        @test Array(adjoint(ft)) ≈ m

        for elt in (Int, UInt32, Float32)
            m = ones(elt, (1, 1))
            ft = to_fusiontensor(m, (g,), (gb,))
            @test eltype(ft) === Float64
            @test Array(ft) ≈ m
        end

        for elt in (ComplexF32, ComplexF64)
            m = ones(elt, (1, 1))
            ft = to_fusiontensor(m, (g,), (gb,))
            @test eltype(ft) === ComplexF64
            @test Array(ft) ≈ m
        end
    end

    @testset "several axes, one block" begin
        g1 = gradedrange([TrivialSector() => 2])
        g2 = gradedrange([TrivialSector() => 3])
        g3 = gradedrange([TrivialSector() => 4])
        g4 = gradedrange([TrivialSector() => 2])
        codomain_legs = (g1, g2)
        domain_legs = dual.((g3, g4))
        t = convert.(Float64, reshape(collect(1:48), (2, 3, 4, 2)))
        ft = to_fusiontensor(t, codomain_legs, domain_legs)
        @test size(data_matrix(ft)) == (6, 8)
        @test blocksize(data_matrix(ft)) == (1, 1)
        @test data_matrix(ft)[Block(1, 1)] ≈ reshape(t, (6, 8))
        @test isnothing(check_sanity(ft))
        @test Array(ft) ≈ t
        @test Array(adjoint(ft)) ≈ permutedims(t, (3, 4, 1, 2))
    end
end

@testset "Abelian FusionTensor" begin
    @testset "trivial matrix" begin
        g = gradedrange([U1(0) => 1])
        gb = dual(g)
        m = ones((1, 1))
        ft = to_fusiontensor(m, (g,), (gb,))
        @test size(data_matrix(ft)) == (1, 1)
        @test blocksize(data_matrix(ft)) == (1, 1)
        @test data_matrix(ft)[1, 1] ≈ 1.0
        @test isnothing(check_sanity(ft))
        @test Array(ft) ≈ m
        @test Array(adjoint(ft)) ≈ m
    end

    @testset "non self-conjugate matrix" begin
        g = gradedrange([U1(1) => 2])
        gb = dual(g)
        m = ones((2, 2))
        ft = to_fusiontensor(m, (g,), (gb,))
        @test size(data_matrix(ft)) == (2, 2)
        @test blocksize(data_matrix(ft)) == (1, 1)
        @test data_matrix(ft)[Block(1, 1)] ≈ m
        @test isnothing(check_sanity(ft))
        @test Array(ft) ≈ m
        @test Array(adjoint(ft)) ≈ m
    end

    @testset "2-block identity" begin
        g = gradedrange([U1(1) => 1, U1(2) => 2])
        codomain_legs = (g,)
        domain_legs = (dual(g),)
        dense = Array{Float64}(LinearAlgebra.I(3))
        ft = to_fusiontensor(dense, codomain_legs, domain_legs)
        @test size(data_matrix(ft)) == (3, 3)
        @test blocksize(data_matrix(ft)) == (2, 2)
        @test data_matrix(ft)[Block(1, 1)] ≈ ones((1, 1))
        @test data_matrix(ft)[Block(2, 2)] ≈ LinearAlgebra.I(2)
        @test data_matrix(ft) ≈ dense
        @test isnothing(check_sanity(ft))
        @test Array(ft) ≈ dense
        @test Array(adjoint(ft)) ≈ adjoint(dense)

        @test_throws BoundsError to_fusiontensor(
            dense, (gradedrange([U1(1) => 1, U1(2) => 3]),), domain_legs
        )
        @test_throws MethodError to_fusiontensor(dense, (g, g), domain_legs)

        ba = BlockedArray(dense, [1, 2], [1, 2])
        @test_throws DomainError to_fusiontensor(
            ba, (gradedrange([U1(1) => 1, U1(2) => 3]),), domain_legs
        )
        @test_throws DomainError to_fusiontensor(ba, (g, g), domain_legs)
        dense[1, 2] = 1  # forbidden
        @test_throws InexactError to_fusiontensor(dense, codomain_legs, domain_legs)
    end

    @testset "several axes, one block" begin
        g1 = gradedrange([U1(1) => 2])
        g2 = gradedrange([U1(2) => 3])
        g3 = gradedrange([U1(3) => 4])
        g4 = gradedrange([U1(0) => 2])
        codomain_legs = (g1, g2)
        domain_legs = dual.((g3, g4))
        t = convert.(Float64, reshape(collect(1:48), (2, 3, 4, 2)))
        ft = to_fusiontensor(t, codomain_legs, domain_legs)
        @test size(data_matrix(ft)) == (6, 8)
        @test blocksize(data_matrix(ft)) == (1, 1)
        @test data_matrix(ft)[Block(1, 1)] ≈ reshape(t, (6, 8))
        @test isnothing(check_sanity(ft))
        @test Array(ft) ≈ t
        @test Array(adjoint(ft)) ≈ permutedims(t, (3, 4, 1, 2))
    end

    @testset "several axes, several blocks" begin
        g1 = gradedrange([U1(1) => 2, U1(2) => 2])
        g2 = gradedrange([U1(2) => 3, U1(3) => 2])
        g3 = gradedrange([U1(3) => 4, U1(4) => 1])
        g4 = gradedrange([U1(0) => 2, U1(2) => 1])
        codomain_legs = (g1, g2)
        domain_legs = dual.((g3, g4))
        dense = zeros((4, 5, 5, 3))
        dense[1:2, 1:3, 1:4, 1:2] .= 1.0
        dense[3:4, 1:3, 5:5, 1:2] .= 2.0
        dense[1:2, 4:5, 5:5, 1:2] .= 3.0
        dense[3:4, 4:5, 1:4, 3:3] .= 4.0
        ft = to_fusiontensor(dense, codomain_legs, domain_legs)
        @test size(data_matrix(ft)) == (20, 15)
        @test blocksize(data_matrix(ft)) == (3, 4)
        @test norm(ft) ≈ norm(dense)
        @test isnothing(check_sanity(ft))
        @test Array(ft) ≈ dense
        @test Array(adjoint(ft)) ≈ permutedims(dense, (3, 4, 1, 2))
    end

    @testset "mixing dual and nondual" begin
        g1 = gradedrange([U1(-1) => 1, U1(0) => 1, U1(1) => 1])
        g2 = gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])
        g3 = gradedrange([U1(0) => 2, U1(1) => 2, U1(3) => 1])
        g4 = gradedrange([U1(-1) => 1, U1(0) => 2, U1(1) => 1])
        codomain_legs = (g1,)
        domain_legs = (dual(g2), dual(g3), g4)
        dense = zeros(ComplexF64, (3, 6, 5, 4))
        dense[2:2, 1:1, 1:2, 2:3] .= 1.0im
        ft = to_fusiontensor(dense, codomain_legs, domain_legs)
        @test size(data_matrix(ft)) == (3, 120)
        @test blocksize(data_matrix(ft)) == (3, 8)
        @test norm(ft) ≈ norm(dense)
        @test isnothing(check_sanity(ft))
        @test Array(ft) ≈ dense
        @test Array(adjoint(ft)) ≈ conj(permutedims(dense, (2, 3, 4, 1)))
    end

    @testset "Less than 2 axes" begin
        g = gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])
        v = zeros((6,))
        v[1] = 1.0

        ft1 = to_fusiontensor(v, (g,), ())
        @test isnothing(check_sanity(ft1))
        @test ndims(ft1) == 1
        @test vec(Array(data_matrix(ft1))) ≈ v
        @test Array(ft1) ≈ v
        @test Array(adjoint(ft1)) ≈ v

        ft2 = to_fusiontensor(v, (), (dual(g),))
        @test isnothing(check_sanity(ft2))
        @test ndims(ft2) == 1
        @test vec(Array(data_matrix(ft2))) ≈ v
        @test Array(ft2) ≈ v
        @test Array(adjoint(ft2)) ≈ v

        ft3 = to_fusiontensor(v, (dual(g),), ())
        @test isnothing(check_sanity(ft3))
        @test Array(ft3) ≈ v
        @test Array(adjoint(ft3)) ≈ v

        ft4 = to_fusiontensor(v, (), (g,))
        @test isnothing(check_sanity(ft4))
        @test Array(ft4) ≈ v
        @test Array(adjoint(ft4)) ≈ v

        if VERSION >= v"1.11"  # https://github.com/JuliaLang/julia/issues/52615
            zerodim = ones(())
            ft = to_fusiontensor(zerodim, (), ())
            @test ft isa FusionTensor
            @test ndims(ft) == 0
            @test isnothing(check_sanity(ft))
            @test size(data_matrix(ft)) == (1, 1)
            @test data_matrix(ft)[1, 1] ≈ 1.0
            @test Array(ft) ≈ zerodim
            @test ndims(Array(ft)) == 0
        end
    end
end

@testset "O(2) FusionTensor" begin
    @testset "trivial matrix" begin
        g = gradedrange([O2(0) => 1])
        gb = dual(g)
        m = ones((1, 1))
        ft = to_fusiontensor(m, (gb,), (g,))
        @test size(data_matrix(ft)) == (1, 1)
        @test blocksize(data_matrix(ft)) == (1, 1)
        @test data_matrix(ft)[1, 1] ≈ 1.0
        @test isnothing(check_sanity(ft))
        @test Array(ft) ≈ m
        @test Array(adjoint(ft)) ≈ m
    end

    @testset "spin 1/2 S.S" begin
        g2 = gradedrange([O2(1 // 2) => 1])
        g2b = dual(g2)

        # identity
        id2 = LinearAlgebra.I((2))
        ft = to_fusiontensor(id2, (g2,), (g2b,))
        @test norm(ft) ≈ √2
        @test isnothing(check_sanity(ft))
        @test Array(ft) ≈ id2
        @test Array(adjoint(ft)) ≈ id2

        # S⋅S
        sds22 = reshape(
            [
                [0.25, 0.0, 0.0, 0.0]
                [0.0, -0.25, 0.5, 0.0]
                [0.0, 0.5, -0.25, 0.0]
                [0.0, 0.0, 0.0, 0.25]
            ],
            (2, 2, 2, 2),
        )
        dense, codomain_legs, domain_legs = sds22, (g2, g2), (g2b, g2b)
        ft = to_fusiontensor(dense, codomain_legs, domain_legs)
        @test norm(ft) ≈ √3 / 2
        @test isnothing(check_sanity(ft))
        @test Array(ft) ≈ sds22
        @test Array(adjoint(ft)) ≈ sds22

        # dual over one spin. This changes the dense coefficients but not the FusionTensor ones
        sds22b = reshape(
            [
                [-0.25, 0.0, 0.0, -0.5]
                [0.0, 0.25, 0.0, 0.0]
                [0.0, 0.0, 0.25, 0.0]
                [-0.5, 0.0, 0.0, -0.25]
            ],
            (2, 2, 2, 2),
        )
        sds22b_codomain_legs = (g2, g2b)
        dense, codomain_legs, domain_legs = sds22b, (g2, g2b), (g2b, g2)
        ftb = to_fusiontensor(dense, codomain_legs, domain_legs)
        @test norm(ftb) ≈ √3 / 2
        @test isnothing(check_sanity(ft))
        @test Array(ftb) ≈ sds22b
        @test Array(adjoint(ftb)) ≈ sds22b

        # no domain axis
        dense, codomain_legs, domain_legs = sds22, (g2, g2, g2b, g2b), ()
        ft = to_fusiontensor(dense, codomain_legs, domain_legs)
        @test isnothing(check_sanity(ft))
        @test Array(ft) ≈ sds22
        @test Array(adjoint(ft)) ≈ sds22

        # no codomain axis
        dense, codomain_legs, domain_legs = sds22, (), (g2, g2, g2b, g2b)
        ft = to_fusiontensor(dense, codomain_legs, domain_legs)
        @test isnothing(check_sanity(ft))
        @test Array(ft) ≈ sds22
        @test Array(adjoint(ft)) ≈ sds22
    end
end

@testset "SU(2) FusionTensor" begin
    @testset "trivial matrix" begin
        g = gradedrange([SU2(0) => 1])
        gb = dual(g)
        m = ones((1, 1))
        ft = to_fusiontensor(m, (gb,), (g,))
        @test size(data_matrix(ft)) == (1, 1)
        @test blocksize(data_matrix(ft)) == (1, 1)
        @test data_matrix(ft)[1, 1] ≈ 1.0
        @test isnothing(check_sanity(ft))
        @test Array(ft) ≈ m
        @test Array(adjoint(ft)) ≈ m
    end

    @testset "spin 1/2 S.S" begin
        g2 = gradedrange([SU2(1 / 2) => 1])
        g2b = dual(g2)

        # identity
        id2 = LinearAlgebra.I((2))
        ft = to_fusiontensor(id2, (g2,), (g2b,))
        @test norm(ft) ≈ √2
        @test isnothing(check_sanity(ft))
        @test Array(ft) ≈ id2
        @test Array(adjoint(ft)) ≈ id2

        # S⋅S
        sds22 = reshape(
            [
                [0.25, 0.0, 0.0, 0.0]
                [0.0, -0.25, 0.5, 0.0]
                [0.0, 0.5, -0.25, 0.0]
                [0.0, 0.0, 0.0, 0.25]
            ],
            (2, 2, 2, 2),
        )
        dense, codomain_legs, domain_legs = sds22, (g2, g2), (g2b, g2b)
        ft = to_fusiontensor(dense, codomain_legs, domain_legs)
        @test norm(ft) ≈ √3 / 2
        @test isnothing(check_sanity(ft))
        @test Array(ft) ≈ sds22
        @test Array(adjoint(ft)) ≈ sds22

        # dual over one spin. This changes the dense coefficients but not the FusionTensor ones
        sds22b = reshape(
            [
                [-0.25, 0.0, 0.0, -0.5]
                [0.0, 0.25, 0.0, 0.0]
                [0.0, 0.0, 0.25, 0.0]
                [-0.5, 0.0, 0.0, -0.25]
            ],
            (2, 2, 2, 2),
        )
        sds22b_codomain_legs = (g2, g2b)
        dense, codomain_legs, domain_legs = sds22b, (g2, g2b), (g2b, g2)
        ftb = to_fusiontensor(dense, codomain_legs, domain_legs)
        @test norm(ftb) ≈ √3 / 2
        @test isnothing(check_sanity(ft))
        @test Array(ftb) ≈ sds22b
        @test Array(adjoint(ftb)) ≈ sds22b

        # no domain axis
        dense, codomain_legs, domain_legs = sds22, (g2b, g2b, g2, g2), ()
        ft = to_fusiontensor(dense, codomain_legs, domain_legs)
        @test isnothing(check_sanity(ft))
        @test Array(ft) ≈ sds22
        @test Array(adjoint(ft)) ≈ sds22

        # no codomain axis
        dense, codomain_legs, domain_legs = sds22, (), (g2b, g2b, g2, g2)
        ft = to_fusiontensor(dense, codomain_legs, domain_legs)
        @test isnothing(check_sanity(ft))
        @test Array(ft) ≈ sds22
        @test Array(adjoint(ft)) ≈ sds22
    end

    @testset "large identity" begin
        g = reduce(tensor_product, (SU2(1 / 2), SU2(1 / 2), SU2(1 / 2)))
        N = 3
        codomain_legs = ntuple(_ -> g, N)
        domain_legs = dual.(codomain_legs)
        d = 8
        dense = reshape(LinearAlgebra.I(d^N), ntuple(_ -> d, 2 * N))
        ft = to_fusiontensor(dense, codomain_legs, domain_legs)
        @test isnothing(check_sanity(ft))
        @test Array(ft) ≈ dense
        @test Array(adjoint(ft)) ≈ dense
    end
end

@testset "U(1)×SU(2) FusionTensor" begin
    for d in 1:6  # any spin dimension
        s = SU2((d - 1) // 2)  # d = 2s+1
        D = d + 1
        tRVB = zeros((d, D, D, D, D))  # tensor RVB SU(2) for spin s
        for i in 1:d
            tRVB[i, i + 1, 1, 1, 1] = 1.0
            tRVB[i, 1, i + 1, 1, 1] = 1.0
            tRVB[i, 1, 1, i + 1, 1] = 1.0
            tRVB[i, 1, 1, 1, i + 1] = 1.0
        end

        gd = gradedrange([SectorProduct(s, U1(3)) => 1])
        codomain_legs = (dual(gd),)
        gD = gradedrange([SectorProduct(SU2(0), U1(1)) => 1, SectorProduct(s, U1(0)) => 1])
        domain_legs = (gD, gD, gD, gD)
        ft = to_fusiontensor(tRVB, codomain_legs, domain_legs)
        @test isnothing(check_sanity(ft))
        @test Array(ft) ≈ tRVB

        # same with NamedTuples
        gd_nt = gradedrange([SectorProduct(; S = s, N = U1(3)) => 1])
        codomain_legs_nt = (dual(gd_nt),)
        gD_nt = gradedrange(
            [
                SectorProduct(; S = SU2(0), N = U1(1)) => 1, SectorProduct(; S = s, N = U1(0)) => 1,
            ]
        )
        domain_legs_nt = (gD_nt, gD_nt, gD_nt, gD_nt)
        ft_nt = to_fusiontensor(tRVB, codomain_legs_nt, domain_legs_nt)
        @test isnothing(check_sanity(ft_nt))
        @test Array(ft_nt) ≈ tRVB
    end
end
