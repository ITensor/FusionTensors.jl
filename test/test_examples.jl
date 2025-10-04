@eval module $(gensym())
using FusionTensors: FusionTensors
using Test: @test, @testset

@testset "examples" begin
    include(joinpath(pkgdir(FusionTensors), "examples", "README.jl"))
end
end
