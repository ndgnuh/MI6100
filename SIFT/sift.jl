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

function compute_progression_sigma(σ::F, num_octaves::Integer,
                                   num_layers::Integer) where {F}
    # Sigma
    num_samples = num_layers + 3
    blur_values = @MVector zeros(F, num_samples)
    blur_values[begin] = σ

    # Compute σ values
    k = 2^(1 / num_layers)
    for i in 1:(num_samples - 1)
        σ_prev = k^(i - 1) * σ
        σ_total = σ_prev * k
        blur_values[i + 1] = sqrt(σ_total^2 - σ_prev^2)
    end

    return blur_values
end

function generate_gaussian_pyramid(base_image,
                                   blur_values,
                                   num_octaves::Integer,
                                   num_layers::Integer)
    T = eltype(base_image)
    base_height, base_width = size(base_image)
    num_samples = num_layers + 3

    gpyr = map(1:num_octaves) do octave
        scale = 2^(octave - 1)
        height = trunc(Int, ceil(base_height / scale))
        width = trunc(Int, ceil(base_width / scale))
        return zeros(T, num_samples, height, width)
    end

    for octave in 1:num_octaves, layer in 1:num_samples
        if octave == layer == 1
            gpyr[octave][layer, :, :] .= base_image
            continue
        end

        if octave != 1 && layer == 1
            base = gpyr[octave - 1][end - 2, :, :]
            gpyr[octave][layer, :, :] .= imresize(base; ratio=1 // 2)
            continue
        end

        #= base = gpyr[octave][layer - 1, :, :] =#
        base = gpyr[octave][1, :, :]
        sigma = blur_values[layer]
        gpyr[octave][layer, :, :] .= imfilter(base, Kernel.gaussian(sigma))
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

        inframe = isinframe((scale, row, col), shape)
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

        # Contrast
        success, new_coord, value = interpolate(dog_octave, coord, grad, hess)
        success = success && abs(value) > 0.001 && test_edge(hess, new_coord)
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
        ori = orientation_bins[scale, (row - radius):(row + radius),
                               (col - radius):(col + radius)]
        mag = orientation_bins[scale, (row - radius):(row + radius),
                               (col - radius):(col + radius)]
        hist = fit(Histogram, view(ori, :); nbins=nbins).weights
        smooth = imfilter(hist, (@SVector [1 // 3, 1 // 3, 1 // 3]))
        dom_weight, dom_orientation = findmax(smooth)
        dom_orientations = findall(smooth) do weight
            return abs((dom_weight - weight) / dom_weight) <= 0.8
        end
        map(dom_orientations) do orientation
            return Keypoint{Float32}(; scale=scale,
                                     row=trunc(Int, row * 2.0f0^(octave_index - 2)),
                                     col=trunc(Int, col * 2.0f0^(octave_index - 2)),
                                     magnitude=4.0f0,
                                     orientation=deg2rad(nbins * orientation))
        end
    end
end

function sift(image_::Matrix, σ::F=1.6, num_layers::Int=3, assume_blur::F=0.5f0) where {F}
    image = convert(Matrix{F}, image_)
    base_image = generate_base_image(image, σ, assume_blur)
    @info typeof(base_image)
    num_octaves = compute_num_octaves(base_image)
    @info "num octave = $num_octaves, $(typeof(num_octaves))"
    blur_values = compute_progression_sigma(σ, num_octaves, num_layers)
    @info "blurs: $blur_values"

    gpyr = generate_gaussian_pyramid(base_image, blur_values,
                                     num_octaves, num_layers)
    @info "gpyr: $(typeof(gpyr))"
    keypoints = mapreduce(union, enumerate(gpyr)) do (octave_index,
                                                      gaussian_octave)
        dog_octave = diff(gaussian_octave; dims=1)
        extrema = (findlocalmaxima(dog_octave; edges=false) ∪
                   findlocalmaxima(dog_octave; edges=false))
        kpts = findkeypoints(dog_octave, extrema)
        kpts_wo = assign_orientations(gaussian_octave, octave_index, kpts)
        return kpts_wo
    end

    return keypoints
end

