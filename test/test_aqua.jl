using Aqua: Aqua
using FusionTensors: FusionTensors
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(FusionTensors)
end
