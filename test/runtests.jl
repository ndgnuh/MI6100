using Test
using SIFT
using ImageCore
using FileIO

box = load(joinpath(@__DIR__, "..", "samples", "box.png"))
keypoints, descriptors = SIFT.sift(box)

@info "Num keypoints $(length(keypoints))"
@test true

@testset "Constant test" begin
    for name in propertynames(SIFT.Constants)
        prop = getproperty(SIFT.Constants, name)
        @test isa(prop, Int) || isa(prop, Float32)
    end
end

