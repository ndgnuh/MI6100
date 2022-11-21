using Test
using SIFT
using ImageCore
using FileIO

@testset "Constant test" begin
    for name in names(SIFT.Constants, all=true)
        prop = getproperty(SIFT.Constants, name)
        if isa(prop, Number)
            @test isa(prop, Int) || isa(prop, Float32)
        end
    end
end

#= box = load(joinpath(@__DIR__, "..", "samples", "box.png")) =#
#= keypoints, descriptors = SIFT.sift(box) =#

#= @info "Num keypoints $(length(keypoints))" =#
#= @test true =#
