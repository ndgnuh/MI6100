module SIFT

using StaticArrays
using Chain: @chain
using ImageTransformations
using LinearAlgebra
using StatsBase
using ImageFiltering
using SplitApplyCombine

	function resize_sampling(I, new_size::NTuple)
	    new_h, new_w = new_size
	    h, w = size(I)
	    r_h, r_w = h / new_h, w / new_w
	    I′ = zeros(eltype(I), new_h, new_w)
	    for i′ in 1:new_h, j′ in 1:new_w
	        i = trunc(Int, r_h * i′ + 0.5)
	        j = trunc(Int, r_w * j′ + 0.5)
	        I′[i′, j′] = I[i, j]
	    end
		return I′
	end
	function resize_sampling(I, ratio::AbstractFloat)
	    new_size = Tuple(trunc(Int, s * ratio) for s in size(I))
	    resize_sampling(I, new_size)
	end
	
include(joinpath(@__DIR__, "constants.jl"))
include(joinpath(@__DIR__, "struct.jl"))
include(joinpath(@__DIR__, "gradient.jl"))
include(joinpath(@__DIR__, "hessian.jl"))
include(joinpath(@__DIR__, "diff.jl"))
include(joinpath(@__DIR__, "draw.jl"))
include(joinpath(@__DIR__, "matching.jl"))
include(joinpath(@__DIR__, "descriptor.jl"))
include(joinpath(@__DIR__, "orientation.jl"))


function gaussian_blur(img, σ...)
    return imfilter(img, KernelFactors.gaussian(σ))
end

"""
    get_octave_pixel_distance(octave_index::Integer; unit_distance=0.5f0)

Return the pixel distance in octave `octave_index`.
In each octave, the image is 0.5x subsampled, therefore the distance
decrease in each octave. The initial octave have distance 0.5 because
the image is doubled in the first octave.
"""
function get_octave_pixel_distance(octave_index::Integer; unit_distance=0.5f0)
    unit_distance * (2^octave_index)
end


function multiply_factor(s)
    return 2.0f0^(1 / s.num_layers)
end

function generate_base_image(image::Matrix{F}, σ::F, assume_blur::F) where {F}
    dσ = sqrt(σ^2 - 4 * assume_blur^2)
    base = imresize(image; ratio=2)
    base = gaussian_blur(base, dσ, dσ)
    return base
end

function compute_num_octaves(sz::Tuple)
    return trunc(Int, round(log(minimum(sz)) / log(2) - 1))
end

function compute_num_octaves(image::Matrix)
    return compute_num_octaves(size(image))
end

function compute_progression_sigma(σ::F,
    num_scales::Integer,
    k::F
) where {F}
    # Sigma
    blur_values = @MVector zeros(F, num_scales)
    @inbounds blur_values[begin] = σ

    # Compute σ values
    for i in 1:(num_scales-1)
        σ_prev = k^(i - 1) * σ
        σ_total = σ_prev * k
        @inbounds blur_values[i+1] = sqrt(σ_total^2 - σ_prev^2)
    end

    return blur_values
end

function compute_absolute_σ(init_σ::F, num_scales, k::F) where {F}
    absolute_σ = Array{F}(undef, num_scales)
    @inbounds absolute_σ[begin] = init_σ
    for idx in 2:num_scales
        @inbounds absolute_σ[idx] = absolute_σ[idx-1] * k
    end
    return absolute_σ
end

function compute_gaussian_pyramid(
    base::Matrix{T},
    init_σ::T,
    num_octaves::Integer,
    num_scales::Integer,
    k::T;
    assume_σ::T=0.5f0
) where {T}
    gpyr = Vector{Array{T,3}}(undef, num_octaves)
    progression_sigmas = compute_progression_sigma(init_σ, num_scales, k)

    for octave_idx in 1:num_octaves
        # Compute octave base
        octave_base = if octave_idx == 1
            first_σ = sqrt(init_σ^2 - 4 * assume_σ^2)
            gaussian_blur(imresize(base, ratio=2), first_σ, first_σ)
        else
            imresize(gpyr[octave_idx-1][end-2, :, :], ratio=1 // 2)
        end
        height, width = size(octave_base)
        @inbounds gpyr[octave_idx] = ones(T, num_scales, height, width)
        @inbounds gpyr[octave_idx][begin, :, :] .= octave_base

        # Compute the scale space associated with the octave
        for scale_idx in 2:num_scales
            σ = progression_sigmas[scale_idx]
            L_prev = gpyr[octave_idx][scale_idx-1, :, :]
            L_next = gaussian_blur(L_prev, σ, σ)
            @inbounds gpyr[octave_idx][scale_idx, :, :] .= L_next
        end
    end
    return gpyr
end


function interpolate(
    dog_octave,
    coord,
    grad,
    hess;
    max_tries=Constants.MAX_INTERPOLATIONS,
    offset_threshold=Constants.OFFSET_THRESH
)
    x = SVector{3,Float32}(i for i in coord.I)
    shape = size(dog_octave)

    success = false
    value = dog_octave[coord]
    scale, row, col = coord.I
    for trial in 1:max_tries
        grad_x = grad[scale, row, col, :]
        hess_x = hess[scale, row, col, :, :]

        dx = -pinv(hess_x) * grad_x
        dvalue = sum(grad_x .* dx) / 2

        # Update
        scale, row, col = trunc.(Int, x + dx)
        value = value + dvalue

        inframe = isinframe((scale, row, col), shape, (0, 1, 1))
        if all(@. abs(dx < offset_threshold)) && inframe
            success = true
            break
        end

        if !inframe
            break
        end
    end
    return success, CartesianIndex(scale, row, col), value
end

function test_edge(
    hess,
    coord;
    edge_threshold=Constants.EDGE_RATIO_THRESH,
    numeric_eps=Constants.EPS
)
    scale, row, col = coord.I
    xy_hessian = hess[scale, row, col, 2:end, 2:end]
    trace = tr(xy_hessian)
    det_ = det(xy_hessian)
    threshold = let r = edge_threshold
        ((r + 1)^2) / r
    end
    return (trace^2) / (det_ + numeric_eps) < threshold
end

function findkeypoints(dog_octave, extrema)
    grad, hess = derivatives(dog_octave)
    Iterators.filter(extrema) do coord
        success, new_coord, value = interpolate(dog_octave, coord, grad, hess)
        success = success && abs(value) > Constants.MAGNITUDE_THRESH && test_edge(hess, new_coord)
        return success
    end
end


function find_extrema(o)
    # First one is (0, 0, 0)
    shifts = Iterators.drop(Iterators.product(-1:1, -1:1, -1:1), 1)
    shape = size(o)
    maxima = ones(Bool, shape)
    minima = ones(Bool, shape)
    for spec in shifts
        shifted = shift(o, spec...)
        @. maxima = maxima && o >= shifted
        @. minima = minima && o <= shifted
    end
    findall(maxima .|| minima)
end


#= function sift(image::Matrix, =#
#=         init_σ=1.6f0, =#
#=         num_octaves=4, =#
#=         num_scales=5, =#
#=         assume_blur=0.5f0 =#
#=         k=sqrt(2)) =#
#=     sift(image, init_σ, num_octaves, num_scales, assume_blur, k) =#
#= end =#
const NUM_SCALES = Constants.SCALES_PER_OCTAVE + Constants.AUXILIARY_SCALES
function sift(image_::Matrix,
    init_σ::F=1.6f0,
    num_octaves::Int=4,
    num_scales::Int=NUM_SCALES,
    assume_blur::F=0.5f0,
    k::F=sqrt(2.0f0)) where {F}
    image = convert(Matrix{F}, image_)

    gpyr = compute_gaussian_pyramid(image, init_σ, num_octaves, num_scales, k)
    @info "gpyr: $(typeof(gpyr))"
    keypoints = mapreduce(union, enumerate(gpyr)) do (octave_index,
        gaussian_octave)
        dog_octave = diff(gaussian_octave; dims=1)
        #= extrema = find_extrema(dog_octave) =#
        extrema = Set([findlocalmaxima(dog_octave); findlocalminima(dog_octave)])
        shape = size(gaussian_octave)

        # Short cut for debug
        #= kpts_wo = map(extrema) do (coord) =#
        #=     scale, row, col = coord.I =#
        #=     Keypoint{Float32}(; scale=scale, =#
        #=                              row=trunc(Int, row * 2.0f0^(octave_index - 2)), =#
        #=                              col=trunc(Int, col * 2.0f0^(octave_index - 2)), =#
        #=                              magnitude=4, =#
        #=                              orientation=0) =#
        #= end =#

        kpts = findkeypoints(dog_octave, extrema)
        kpts_wo = assign_orientations(gaussian_octave, octave_index, kpts; init_σ=init_σ, num_scales=num_scales, k=k)
        return kpts_wo
    end

    descriptors = map(keypoints) do kpt
        octave_index = kpt.octave
        L = gpyr[octave_index]
        compute_descriptor(kpt, L, octave_index)
    end

    return keypoints, descriptors
end


function get_gradient_props(gpyr, num_octaves, num_scales)
    iter = Iterators.product(1:num_octaves, 1:num_scales)
    data = map(iter) do (octave_index, scale_index)
        L = view(gpyr[octave_index], scale_index, :, :)
        dr = shift(L, 1, 0) - shift(L, -1, 0)
        dc = shift(L, 0, 1) - shift(L, 0, -1)
        mag = @. sqrt(dr^2 + dc^2)
        ori = @. atan(dc, dr) % (2 * pi)
        mag, ori
    end
    return invert(data)
end
#= if PROGRAM_FILE == @__FILE__ =#
#=     @info 123 =#
#= end =#


end
end
