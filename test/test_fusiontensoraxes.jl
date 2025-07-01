using Test: @test, @test_throws, @testset

using TensorProducts: ⊗
using BlockArrays: Block, blockedrange, blocklength, blocklengths, blocks
using TensorAlgebra: BlockedTuple, trivial_axis, tuplemortar

using FusionTensors:
  FusionTensorAxes,
  ndims_domain,
  ndims_codomain,
  codomain,
  domain,
  fused_codomain,
  fused_domain,
  promote_sector_type,
  promote_sectors
using GradedArrays:
  ×,
  U1,
  SectorProduct,
  TrivialSector,
  SU2,
  checkspaces,
  dual,
  gradedrange,
  sector_type,
  space_isequal

@testset "misc FusionTensors.jl" begin
  g1 = gradedrange([U1(0) => 1])
  @test promote_sector_type(U1(1), U1(1)) === typeof(U1(1))
  @test promote_sector_type(g1, U1(1)) === typeof(U1(1))
  @test promote_sector_type(g1, g1) === typeof(U1(1))
  @test promote_sector_type((g1, g1)) === typeof(U1(1))

  sNS = SectorProduct(; N=U1(1), S=SU2(1 / 2))
  gN = gradedrange([(; N=U1(1)) => 1])
  gS = gradedrange([(; S=SU2(1 / 2)) => 1])
  @test promote_sector_type(gN, gS) == typeof(sNS)

  @test promote_sectors((gN, gS)) isa NTuple{2,typeof(gradedrange([sNS => 1]))}
end

@testset "FusionTensorAxes" begin
  s2 = SU2(1//2)
  g2 = gradedrange([s2 => 1])
  g2b = dual(g2)

  bt = tuplemortar(((g2, g2), (g2b, g2b)))
  fta = FusionTensorAxes(bt)

  @test fta isa FusionTensorAxes
  @test BlockedTuple(fta) == bt

  @test Tuple(fta) == Tuple(bt)
  @test space_isequal(only(axes(fta)), blockedrange([2, 2]))
  @test iterate(fta) == (g2, 2)
  @test iterate(fta, 1) == (g2, 2)
  @test iterate(fta, 2) == (g2, 3)
  @test iterate(fta, 3) == (g2b, 4)
  @test iterate(fta, 4) == (g2b, 5)
  @test isnothing(iterate(fta, 5))

  @test length(fta) == 4
  @test space_isequal(fta[1], g2)
  @test space_isequal(fta[2], g2)
  @test space_isequal(fta[3], g2b)
  @test space_isequal(fta[4], g2b)
  @test length(fta[Block(1)]) == 2
  @test all(map(r -> space_isequal(r, g2), fta[Block(1)]))
  @test length(fta[Block(2)]) == 2
  @test all(map(r -> space_isequal(r, g2b), fta[Block(2)]))
  @test length.(fta) == tuplemortar(((2, 2), (2, 2)))

  @test blocklength(fta) == 2
  @test blocklengths(fta) == (2, 2)
  @test blocks(fta) == blocks(bt)

  @test sector_type(fta) === sector_type(g2)
  @test length(codomain(fta)) == 2
  @test all(map(r -> space_isequal(r, g2), codomain(fta)))
  @test length(domain(fta)) == 2
  @test all(map(r -> space_isequal(r, g2b), domain(fta)))
  @test space_isequal(fused_codomain(fta), g2 ⊗ g2)
  @test space_isequal(fused_domain(fta), dual(g2 ⊗ g2))
  @test space_isequal(trivial_axis(fta), trivial_axis(typeof(s2)))

  @test fta == fta
  @test copy(fta) == fta
  @test deepcopy(fta) == fta
  @test fta != FusionTensorAxes(tuplemortar(((g2, g2), (g2b, g2))))
  @test fta != FusionTensorAxes(tuplemortar(((g2, g2, g2b), (g2b,))))

  @test fta == FusionTensorAxes((g2, g2), (g2b, g2b))
  @test checkspaces(fta, fta)
  @test_throws ArgumentError checkspaces(
    fta, FusionTensorAxes(tuplemortar(((g2, g2), (g2b, g2))))
  )
end

@testset "Empty FusionTensorAxes" begin
  fta = FusionTensorAxes(tuplemortar(((), ())))
  @test fta isa FusionTensorAxes

  @test length(fta) == 0
  @test isempty(fta)
  @test blocklength(fta) == 2
  @test blocklengths(fta) == (0, 0)
  @test sector_type(fta) == TrivialSector

  @test codomain(fta) == ()
  @test space_isequal(fused_codomain(fta), trivial_axis(TrivialSector))
  @test domain(fta) == ()
  @test space_isequal(fused_domain(fta), dual(trivial_axis(TrivialSector)))
end
