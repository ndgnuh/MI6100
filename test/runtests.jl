using Test
using SIFT
using ImageCore
using FileIO
using Chain

@testset "Constant test" begin
    for name in names(SIFT.Constants, all=true)
        prop = getproperty(SIFT.Constants, name)
        if isa(prop, Number)
            @test isa(prop, Int) || isa(prop, Float32)
        end
    end
end



image = @chain begin
    load(joinpath(@__DIR__, "..", "samples", "box.png"))
    convert(Matrix{Float32}, _)
end
@time keypoints, descriptors = SIFT.sift(image)
@time keypoints, descriptors = SIFT.sift(image)
#= gpyr = SIFT.compute_gaussian_pyramid( =#
#=     image, =#
#=     1.6f0, =#
#=     4, =#
#=     3, =#
#=     sqrt(2.0f0) =#
#= ) =#

#= orientations, magnitudes = SIFT.get_gradient_props(gpyr, 4, 3) =#

#= @info "Num keypoints $(length(keypoints))" =#
#= @test true =#
