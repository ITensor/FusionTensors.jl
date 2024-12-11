@eval module $(gensym())
using Test: @test, @testset
using LinearAlgebra: I

using BlockArrays: BlockArrays

using FusionTensors: fusion_tree_tensors
using GradedUnitRanges: blocklabels, fusion_product
using SymmetrySectors: SectorProduct, SU, SU2, TrivialSector, U1, Z, quantum_dimension

@testset "Trivial fusion trees" begin
  tree_irreps_pairs1 = fusion_tree_tensors((), ())
  @test first.(tree_irreps_pairs1) == [TrivialSector()]
  @test last.(tree_irreps_pairs1) == [ones((1, 1))]

  tree_irreps_pairs2 = fusion_tree_tensors(
    (TrivialSector(), TrivialSector()), (false, false)
  )
  @test first.(tree_irreps_pairs2) == [TrivialSector()]
  @test last.(tree_irreps_pairs2) == [ones((1, 1, 1, 1))]

  tree_irreps_pairs3 = fusion_tree_tensors(
    (TrivialSector(), TrivialSector(), TrivialSector()), (false, true, true)
  )
  @test first.(tree_irreps_pairs3) == [TrivialSector()]
  @test last.(tree_irreps_pairs3) == [ones((1, 1, 1, 1, 1))]
end

@testset "Abelian fusion trees" begin
  tree_irreps_pairs1 = fusion_tree_tensors((U1(0), U1(0)), (false, false))
  @test first.(tree_irreps_pairs1) == [U1(0)]
  @test last.(tree_irreps_pairs1) == [ones((1, 1, 1, 1))]

  tree_irreps_pairs2 = fusion_tree_tensors((U1(1), U1(1), U1(-2)), (false, false, false))
  @test first.(tree_irreps_pairs2) == [U1(0)]
  @test last.(tree_irreps_pairs2) == [ones((1, 1, 1, 1, 1))]

  s1 = SectorProduct(Z{2}(1), U1(0))
  s2 = SectorProduct(Z{2}(1), U1(1))
  tree_irreps_z2u1 = fusion_tree_tensors((s1, s2), (false, false))
  @test first.(tree_irreps_z2u1) == [SectorProduct(Z{2}(0), U1(1))]
  @test last.(tree_irreps_z2u1) == [ones((1, 1, 1, 1))]

  s3 = SectorProduct(; A=Z{2}(1), B=U1(0))
  s4 = SectorProduct(; A=Z{2}(1), B=U1(1))
  tree_irreps_nt = fusion_tree_tensors((s3, s4), (false, false))
  @test first.(tree_irreps_nt) == [SectorProduct(; A=Z{2}(0), B=U1(1))]
  @test last.(tree_irreps_nt) == [ones((1, 1, 1, 1))]
end

@testset "SU(2) fusion trees" begin
  tree_irreps_pairs1 = fusion_tree_tensors((SU2(0), SU2(0), SU2(0)), (false, false, false))
  @test first.(tree_irreps_pairs1) == [SU2(0)]
  @test last.(tree_irreps_pairs1) == [ones((1, 1, 1, 1, 1))]

  tree_irreps_pairs2 = fusion_tree_tensors((SU2(1 / 2),), (false,))
  @test first.(tree_irreps_pairs2) == [SU2(1 / 2)]
  @test last.(tree_irreps_pairs2) == [reshape(I(2), (2, 2, 1))]

  tree_irreps_pairs3 = fusion_tree_tensors((SU2(1 / 2),), (true,))
  @test first.(tree_irreps_pairs3) == [SU2(1 / 2)]
  @test last.(tree_irreps_pairs3) ≈ [reshape([0, 1, -1, 0], (2, 2, 1))]

  tree_irreps_pairs4 = fusion_tree_tensors(
    (SU2(1 / 2), SU2(1 / 2), SU2(1 / 2)), (false, false, false)
  )
  @test first.(tree_irreps_pairs4) == [SU2(1 / 2), SU2(3 / 2)]
  @test size.(last.(tree_irreps_pairs4)) == [(2, 2, 2, 2, 2), (2, 2, 2, 4, 1)]
end

@testset "SectorProduct fusion trees" begin
  hole = SectorProduct(; N=U1(1), S=SU2(0))
  s12 = SectorProduct(; N=U1(0), S=SU2(1 / 2))
  tree_irreps_pairs1 = fusion_tree_tensors(
    (hole, hole, hole, s12), (false, false, false, false)
  )
  @test first.(tree_irreps_pairs1) == [SectorProduct(; N=U1(3), S=SU2(1 / 2))]
  @test last.(tree_irreps_pairs1) == [reshape(I(2), (1, 1, 1, 2, 2, 1))]

  s3 = SectorProduct(SU((1, 0)), SU((1,)), U1(1))
  irreps = (s3, s3, s3)
  arrows = (false, false, false)
  tree_irreps_pairs2 = fusion_tree_tensors(irreps, arrows)
  tree_irreps, trees = first.(tree_irreps_pairs2), last.(tree_irreps_pairs2)
  rep = fusion_product(irreps...)
  @test blocklabels(rep) == tree_irreps
  @test quantum_dimension.(tree_irreps) == size.(trees, 4)
  @test BlockArrays.blocklengths(rep) == size.(trees, 5)

  s_nt = SectorProduct(; A=SU((1, 0)), B=SU((1,)), C=U1(1))
  irreps_nt = (s_nt, s_nt, s_nt)
  tree_irreps_nt_pairs = fusion_tree_tensors(irreps_nt, arrows)
  tree_irreps_nt, trees_nt = first.(tree_irreps_nt_pairs), last.(tree_irreps_nt_pairs)
  rep_nt = reduce(fusion_product, irreps_nt)
  @test blocklabels(rep_nt) == tree_irreps_nt
  @test quantum_dimension.(tree_irreps_nt) == size.(trees_nt, 4)
  @test BlockArrays.blocklengths(rep_nt) == size.(trees_nt, 5)
end
end
