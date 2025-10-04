using Test: @test, @testset
using LinearAlgebra: I

using BlockArrays: BlockArrays

using FusionTensors:
    SectorFusionTree,
    arrows,
    branch_sectors,
    build_trees,
    fusiontree_eltype,
    leaves,
    outer_multiplicity_indices,
    root_sector
using GradedArrays:
    ×, SectorProduct, SU, SU2, TrivialSector, arguments, dual, flip, gradedrange, sector_type

@testset "Trivial fusion trees" begin
    q = TrivialSector()
    f = SectorFusionTree{TrivialSector}()
    @test arrows(f) == ()
    @test leaves(f) == ()
    @test root_sector(f) == q
    @test branch_sectors(f) == ()
    @test sector_type(f) == TrivialSector
    @test outer_multiplicity_indices(f) == ()
    @test convert(Array, f) ≈ ones((1,))

    f = only(build_trees((q,), (true,)))
    @test arrows(f) == (true,)
    @test leaves(f) == (q,)
    @test root_sector(f) == q
    @test branch_sectors(f) == ()
    @test outer_multiplicity_indices(f) == ()
    @test convert(Array, f) ≈ ones((1, 1))

    f = only(build_trees((q, q), (true, false)))
    @test arrows(f) == (true, false)
    @test leaves(f) == (q, q)
    @test root_sector(f) == q
    @test branch_sectors(f) == (q,)
    @test outer_multiplicity_indices(f) == (1,)
    @test convert(Array, f) ≈ ones((1, 1, 1))

    @test fusiontree_eltype(sector_type(f)) === eltype(convert(Array, f))
end

@testset "SU(2) SectorFusionTree" begin
    j2 = SU2(1 // 2)

    f = only(build_trees((j2,), (false,)))
    @test arrows(f) == (false,)
    @test leaves(f) == (j2,)
    @test root_sector(f) == j2
    @test branch_sectors(f) == ()
    @test outer_multiplicity_indices(f) == ()
    @test sector_type(f) == typeof(j2)
    @test convert(Array, f) ≈ I(2)
    @test fusiontree_eltype(sector_type(f)) === eltype(convert(Array, f))

    f = only(build_trees((j2,), (true,)))
    @test arrows(f) == (true,)
    @test convert(Array, f) ≈ [0 -1; 1 0]

    trees = build_trees((j2, j2), (false, false))
    @test length(trees) == 2
    f1 = first(trees)
    @test root_sector(f1) == SU2(0)
    @test branch_sectors(f1) == (SU2(1 // 2),)
    @test outer_multiplicity_indices(f1) == (1,)
    @test convert(Array, f1) ≈ [0 1 / sqrt(2); -1 / sqrt(2) 0]

    f3 = last(trees)
    @test root_sector(f3) == SU2(1)
    @test branch_sectors(f3) == (SU2(1 // 2),)
    @test outer_multiplicity_indices(f3) == (1,)
    t = zeros((2, 2, 3))
    t[1, 1, 1] = 1
    t[1, 2, 2] = 1 / sqrt(2)
    t[2, 1, 2] = 1 / sqrt(2)
    t[2, 2, 3] = 1
    @test convert(Array, f3) ≈ t

    trees = build_trees((j2, j2, j2), (false, false, false))
    @test length(trees) == 3
    f12, f32, f34 = trees
    @test f12 < f32 < f34
    @test root_sector(f12) == SU2(1 // 2)
    @test root_sector(f32) == SU2(1 // 2)
    @test root_sector(f34) == SU2(3 // 2)

    @test branch_sectors(f12) == (SU2(1 // 2), SU2(0))
    @test branch_sectors(f32) == (SU2(1 // 2), SU2(1))
    @test branch_sectors(f34) == (SU2(1 // 2), SU2(1))
end

@testset "SU(3) SectorFusionTree" begin
    # convention: irreps are already dualed if needed, arrows do not affect them. They only
    # affect the basis on which the tree projects for self-dual irreps.
    f3 = SU{3}((1, 0))
    c3 = dual(f3)

    trees = build_trees((f3, f3), (false, false))
    @test root_sector.(trees) == [SU{3}((1, 1)), SU{3}((2, 0))]

    trees = build_trees((f3, f3), (true, false))
    @test root_sector.(trees) == [SU{3}((1, 1)), SU{3}((2, 0))]

    trees = build_trees((f3, f3), (false, true))
    @test root_sector.(trees) == [SU{3}((1, 1)), SU{3}((2, 0))]

    trees = build_trees((f3, f3), (true, true))
    @test root_sector.(trees) == [SU{3}((1, 1)), SU{3}((2, 0))]

    trees = build_trees((f3, c3), (false, false))
    @test root_sector.(trees) == [SU{3}((0, 0)), SU{3}((2, 1))]

    trees = build_trees((c3, c3), (false, false))
    @test root_sector.(trees) == [SU{3}((1, 0)), SU{3}((2, 2))]

    # test GradedUnitRange interface
    g = gradedrange([SU{3}((1, 0)) => 1])
    trees = build_trees(g, g)
    @test root_sector.(trees) == [SU{3}((1, 1)), SU{3}((2, 0))]
    trees = build_trees(g, flip(g))
    @test root_sector.(trees) == [SU{3}((1, 1)), SU{3}((2, 0))]
    trees = build_trees(g, flip(dual(g)))
    @test root_sector.(trees) == [SU{3}((0, 0)), SU{3}((2, 1))]
    trees = build_trees(g, dual(g))
    @test root_sector.(trees) == [SU{3}((0, 0)), SU{3}((2, 1))]

    # test outer outer_multiplicity > 1
    a8 = SU{3}((2, 1))
    trees = build_trees((a8, a8), (false, false))
    @test length(trees) == 6
    f = first(trees)
    @test root_sector(f) == SU{3}((0, 0))
    @test sector_type(f) == typeof(a8)

    f8a = trees[2]
    f8b = trees[3]
    @test root_sector(f8a) == a8
    @test root_sector(f8b) == a8
    @test branch_sectors(f8a) == (a8,)
    @test branch_sectors(f8b) == (a8,)
    @test outer_multiplicity_indices(f8a) == (1,)
    @test outer_multiplicity_indices(f8b) == (2,)
end

@testset "SU(2)×SU(3) SectorFusionTree" begin
    j2 = SU2(1 // 2)
    a8 = SU{3}((2, 1))
    s = j2 × a8
    trees = build_trees((s, s), (false, false))
    @test length(trees) == 12
    f = first(trees)
    @test sector_type(f) == typeof(s)
    argument_trees = arguments(f)
    @test length(argument_trees) == 2
    f2, f3 = argument_trees
    @test sector_type(f2) == typeof(j2)
    @test sector_type(f3) == typeof(a8)
    @test f == f2 × f3
end
