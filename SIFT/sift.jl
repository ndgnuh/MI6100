using Accessors
using Memoize
using Base: IdDict
using Parameters
using UnPack
using StaticArrays
using Chain: @chain
using ImageTransformations
using LinearAlgebra
using StatsBase
using ImageFiltering

include(joinpath(@__DIR__, "struct.jl"))
include(joinpath(@__DIR__, "gradient.jl"))
include(joinpath(@__DIR__, "hessian.jl"))
include(joinpath(@__DIR__, "diff.jl"))

function multiply_factor(s)
    return 2.0f0^(1 / s.num_layers)
end

function generate_base_image(image::Matrix{F}, σ::F, assume_blur::F) where {F}
    dsigma = sqrt(σ^2 - 4 * assume_blur^2)
    base = imresize(image; ratio=2)
    base = imfilter(base, Kernel.gaussian(dsigma))
    return base
end

function compute_num_octaves(sz::Tuple)
    return trunc(Int, round(log(minimum(sz)) / log(2) - 1))
end

function compute_num_octaves(image::Matrix)
    return compute_num_octaves(size(image))
end

function compute_progression_sigma(σ::F, num_scales::Integer, num_aux_scales::Integer) where {F}
    # Sigma
    blur_values = @MVector zeros(F, num_scales + num_aux_scales)
    blur_values[begin] = σ

    # Compute σ values
    k = 2^(1 / num_scales)
    for i in 1:(num_scales+num_aux_scales-1)
        σ_prev = k^(i - 1) * σ
        σ_total = σ_prev * k
        blur_values[i+1] = sqrt(σ_total^2 - σ_prev^2)
    end

    return blur_values
end

function compute_gaussian_pyramid(base::Matrix{T}, init_σ::T, num_octaves::Integer,
    num_scales::Integer, num_aux_scales::Integer=3;
    assume_σ::T=0.5f0) where {T}
    num_total_scales = num_scales + num_aux_scales
    gpyr = Vector{Array{T,3}}(undef, num_octaves)
    progression_sigmas = compute_progression_sigma(init_σ, num_scales, num_aux_scales)
    @info progression_sigmas

    for octave_idx in 1:num_octaves
        # Compute octave base
        octave_base = if octave_idx == 1
            first_σ = sqrt(init_σ^2 - 4 * assume_σ^2)
            imfilter(imresize(base, ratio=2), Kernel.gaussian(first_σ))
        else
            imresize(gpyr[octave_idx-1][end-2, :, :], ratio=1 // 2)
        end
        height, width = size(octave_base)
        gpyr[octave_idx] = ones(T, num_total_scales, height, width)
        gpyr[octave_idx][begin, :, :] .= octave_base

        # Compute the scale space associated with the octave
        for scale_idx in 2:num_total_scales
            σ = progression_sigmas[scale_idx]
            L_prev = gpyr[octave_idx][scale_idx-1, :, :]
            L_next = imfilter(L_prev, Kernel.gaussian(σ))
            gpyr[octave_idx][scale_idx, :, :] .= L_next
        end
    end
    return gpyr
end


function interpolate(dog_octave, coord, grad, hess, max_tries=10)
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
        if all(@. abs(dx < 0.5)) && inframe
            success = true
            break
        end

        if !inframe
            break
        end
    end
    return success, CartesianIndex(scale, row, col), value
end

function test_edge(hess, coord, r=10)
    e = eps(eltype(hess))
    scale, row, col = coord.I
    xy_hessian = hess[scale, row, col, 2:end, 2:end]
    trace = tr(xy_hessian)
    det_ = det(xy_hessian) + e
    threshold = ((r + 1)^2) / r
    return (trace^2) / (det_ + e) < threshold
end

function findkeypoints(dog_octave, extrema)
    grad, hess = derivatives(dog_octave)
    Iterators.filter(extrema) do coord
        success, new_coord, value = interpolate(dog_octave, coord, grad, hess)
        success = success && abs(value) > 0.03f0 && test_edge(hess, new_coord)
        return success
    end
end

function assign_orientations(gaussian_octave, octave_index, coords, nbins::Int=36)
    magnitudes, orientations = let o = gaussian_octave
        dr = shift(o, 0, 1, 0) - shift(o, 0, -1, 0)
        dc = shift(o, 0, 0, 1) - shift(o, 0, 0, -1)
        mag = @. sqrt(dr^2 + dc^2)
        ori = @. atan(dc, dr) % (2 * pi)
        mag, ori
    end
    shape = size(gaussian_octave)

    # TODO: calculate radius
    radius = 4
    orientation_bins = padarray((@. trunc(Int, nbins / 2 / pi * orientations)),
        Pad(:reflect, radius, radius, radius))
    return mapreduce(union, coords; init=NamedTuple[]) do coord
        scale, row, col = coord.I
        ori = orientation_bins[scale, (row-radius):(row+radius),
            (col-radius):(col+radius)]
        mag = orientation_bins[scale, (row-radius):(row+radius),
            (col-radius):(col+radius)]
        hist = fit(Histogram, view(ori, :); nbins=nbins).weights
        smooth = imfilter(hist, (@SVector [1 // 3, 1 // 3, 1 // 3]))
        dom_weight, dom_orientation = findmax(smooth)
        dom_orientations = findall(smooth) do weight
            return abs((dom_weight - weight) / dom_weight) <= 0.8
        end
        Iterators.map(dom_orientations) do orientation
            return Keypoint{Float32}(; scale=scale,
                row=trunc(Int, row * 2.0f0^(octave_index - 2)),
                col=trunc(Int, col * 2.0f0^(octave_index - 2)),
                magnitude=4,
                orientation=deg2rad(nbins * orientation))
        end
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

function sift(image_::Matrix, init_σ::F=1.6, num_scales::Int=3, assume_blur::F=0.5f0) where {F}
    image = convert(Matrix{F}, image_)

    gpyr = compute_gaussian_pyramid(image, init_σ,
        compute_num_octaves(image),
        num_scales)
    @info "gpyr: $(typeof(gpyr))"
    keypoints = mapreduce(union, enumerate(gpyr)) do (octave_index,
        gaussian_octave)
        dog_octave = diff(gaussian_octave; dims=1)
        #= extrema = find_extrema(dog_octave) =#
        extrema = [findlocalmaxima(dog_octave); findlocalminima(dog_octave)]
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
        kpts_wo = assign_orientations(gaussian_octave, octave_index, kpts)
        return kpts_wo
    end

    return keypoints
end

