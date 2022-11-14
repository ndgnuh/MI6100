using Accessors
using Memoize
using Base: IdDict
using Parameters
using UnPack
using StaticArrays
using Chain: @chain
using ImageTransformations

macro lazy_context(dispatch_e, T)
    dispatch = eval(dispatch_e)
    quote
        function Base.getproperty(ctx::$(esc(T)), prop::Symbol)
            value = getfield(ctx, prop)
            if isnothing(value) && prop in keys($(esc(dispatch_e)))
                setproperty!(ctx, prop, $(esc(dispatch_e))[prop](ctx))
            end
            return getfield(ctx, prop)
        end
    end
end

include(joinpath(@__DIR__, "gradient.jl"))
include(joinpath(@__DIR__, "hessian.jl"))

function multiply_factor(s)
    return 2.0f0^(1 / s.num_layers)
end

function generate_base_image(image::Matrix, σ, assume_blur)
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

function compute_blur_values(σ::F, num_octaves::Integer, num_layers::Integer) where {F}
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
            base = gpyr[octave - 1][end - 3, :, :]
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

function generate_dog_pyramid(gpyr::Vector)
    return map(gpyr) do Ls
        return diff(Ls; dims=1)
    end
end

function localize_keypoint(dpyr, gradient, hessian, σ, octave, layer, row, col)
    dog = dpyr[octave]
    grad = gradient[octave]
    hess = hessian[octave]
    mlayer::Int, mrow::Int, mcol::Int = size(grad.x)

    # Candidate property
    low_contrast::Bool = true
    on_edge::Bool = true
    outside::Bool = false

    # Localize via quadratic fit
    x = @MVector Float32[layer, row, col]
     x̂ = @MVector zeros(Float32, 3)
    converge = false
    for i in 1:10
        layer, row, col = trunc.(Int, x)

        # Check if the coordinate is inside the image
        if (layer < 2 || layer > mlayer - 1
            || row < 2 || row > mrow - 1
            || col < 2 || col > mcol - 1)
            outside = true
            break
        end

         # Solve for x̂
        g = grad(layer, row, col)
        h = hess(layer, row, col)
        if det(h) == 0
            break
        end
         x̂ .= -inv(h) * g

        # Check if the changes is small
         if any(@. abs(x̂) <= 0.5)
            converge = true
            break
        end

        # Update coordinate
         x .= x + x̂
    end

    # True coordinate (maybe)
    layer, row, col = trunc.(Int, x)

    # More check if localized
    contrast::Float32 = 0
    if converge
        # Calculate contrast
        g = grad(layer, row, col)
         contrast = dog[layer, row, col] + sum(g .* x̂) / 2
        low_contrast = abs(contrast) * mlayer < 0.04

        # Calcuate edge response
        h = @view hess(layer, row, col)[2:3, 2:3]
        det_h = det(h)
        tr_h = tr(h)
        r = 10
        on_edge = det_h <= 0 || tr_h^2 * r >= det_h * (r + 1)^2
    end

    # Calculate angle if
    # keypoint is high contrast and
    # keypoint is not on any edge
    sigma = 1
    kptsize = σ * (2^(layer - 1 / (mlayer - 3)) * (2^(octave - 1)))
    #= kptsize = 3 =#

    return (outside=outside,
            size=kptsize,
            converge=converge,
            contrast=abs(contrast),
            low_contrast=low_contrast,
            on_edge=on_edge,
            angle=0,
            octave=octave,
            layer=layer,
            row=trunc(Int, row),#  * 2.0^(octave - 2)
            col=trunc(Int, col))#  * 2.0^(octave - 2) # row = row, col=col
end

function find_scale_extremas(dpyr, σ)
    num_octaves = length(dpyr)

    dog_hess = Dict(octave => Hessian(dpyr[octave])
                    for octave in 1:num_octaves)
    dog_grad = Dict(octave => Gradient(dpyr[octave])
                    for octave in 1:num_octaves)

    return mapreduce(union, enumerate(dpyr)) do (octave, dog)
        map(findlocalmaxima(dog) ∪ findlocalmaxima(dog)) do idx
            layer, row, col = idx.I
            return localize_keypoint(dpyr, dog_grad, dog_hess, σ, octave, layer, row, col)
        end
    end
end

function compute_keypoints_with_orientation(grad, kpt, nbins=36)
    octave = kpt.octave
    layer = kpt.layer
    row = kpt.row
    col = kpt.col
    histogram = @MVector zeros(nbins)
    rad = trunc(Int, kpt.size) * 3

    surround_idx = ((kpt.row + i, kpt.col + j)
                    for i in (-rad):rad, j in (-rad):rad
                    if i^2 + j^2 <= rad^2)
    for (row, col) in surround_idx
        m = magnitude_at(grad[octave, layer], row, col)
        rad = angle_at(grad[octave, layer], row, col)
        deg = rad2deg(rad)
        bin = (360 + trunc(Int, deg)) % nbins + 1
        histogram[bin] += m
    end
    histogram /= sum(histogram)
    #= @info histogram =#

    angle, _ = findmax(histogram)
    angle = angle * 10
    @set kpt.angle = deg2rad(angle)
    @set kpt.size = magnitude_at(grad[octave, layer], row, col)
    return SVector(kpt)
end

function compute_keypoints_with_orientation(gpyr::Vector, keypoints)
    num_octaves = length(gpyr)
    num_samples = size(gpyr[1])[1]
    grad = SMatrix{num_octaves,num_samples}(Gradient(gpyr[octave][layer, :, :])
                                            for octave in 1:num_octaves,
                                                layer in 1:num_samples)
    @info grad
    keypoints2 = mapreduce(union, keypoints) do kpt
        return compute_keypoints_with_orientation(grad, kpt)
    end
    return keypoints2
end

function sift(image, σ::F=1.6, num_layers=3, assume_blur=0.5) where {F}
    base_image = generate_base_image(convert(Matrix{F}, image), σ, assume_blur)
    @info typeof(base_image)
    num_octaves = compute_num_octaves(base_image)
    @info "num octave = $num_octaves"
    blur_values = compute_blur_values(σ, num_octaves, num_layers)
    @info "blurs: $blur_values"
    gpyr = generate_gaussian_pyramid(base_image, blur_values, num_octaves, num_layers)
    @info "gpyr: $(size(gpyr))"
    dpyr = generate_dog_pyramid(gpyr)
    @info "dpyr: $(size(gpyr))"
    base_keypoints = find_scale_extremas(dpyr, σ)
    @info "base keypoints $(length(base_keypoints)), $(eltype(base_keypoints))"
    filtered_keypoints = filter(base_keypoints) do kpt
        return kpt.converge && !kpt.outside && !kpt.low_contrast && !kpt.on_edge
    end
    @info "filtered keypoints $(length(filtered_keypoints)), $(eltype(filtered_keypoints))"
    keypoints = compute_keypoints_with_orientation(gpyr, filtered_keypoints)
    @info "keypoints with orientation: $(length(keypoints))"
    #= keypoints_ = compute_keypoints_descriptor(gpyr, keypoints) =#
    return keypoints
end

## THIS MUST BE RUN AT THE END OF THE MODULE

Maybe{T} = Union{Nothing,T}

@with_kw mutable struct SIFT
    sigma = 1.6
    num_layers = 3
    assumed_blur = 0.5
    base_image = nothing
    k = nothing
    sigmas = nothing
    gpyr = nothing
    dpyr = nothing
    keypoints = nothing
    localized_keypoints = nothing
    d_grad = nothing
    d_hess = nothing
    g_grad = nothing
    g_hess = nothing
end

@with_kw struct SIFTContext{F}
    sift::SIFT
    image::Matrix{F}
    base_image::Matrix{F}
    g_grad::Gradient{F,3}
    g_hess::Hessian{F,3}
    d_grad::Gradient{F,3}
    d_hess::Hessian{F,3}
    g_pyr::IdDict{Int,Array{F,3}} = IdDict()
    d_pyr::IdDict{Int,Array{F,3}} = IdDict()
end

@with_kw struct Keypoint
    octave::Int
    layer::Int
    row::Int
    col::Int
    size::Maybe{Float32} = nothing
    angle::Maybe{Float32} = nothing
    low_contrast::Maybe{Bool} = nothing
    edge::Maybe{Bool} = nothing
    localized::Maybe{Bool} = false
end

function multiply_factor(s::SIFT)
    return 2^(1 / s.num_layers)
end

function localize_keypoint(s, octave, row, col, layer) end

function compute_keypoints(s::SIFT)
    mapreduce(union, enumerate(s.dpyr)) do (octave, dog)
        @chain begin
            findlocalmaxima(dog) ∪ findlocalminima(dog)
            map(_) do kpt
                return Keypoint(; row=kpt[2], col=kpt[3], layer=kpt[1], octave=octave)
            end
        end
    end
end

include(joinpath(@__DIR__, "diff.jl"))
function compute_localized_keypoints(s::SIFT)
    kpts = s.keypoints
    map(kpts) do kpt
        @unpack row, col, octave, layer = kpt
        num_layers, height, width = size(s.dpyr[octave])
        x = Float32[layer, row, col]
        num_attempts = 10
        converge = false
        for i in 1:num_attempts
            layer, row, col = trunc.(Int, x)
            if (row < 2 ||
                col < 2 ||
                layer < 2 ||
                row > height - 1 ||
                col > width - 1 ||
                layer > num_layers - 1)
                break
            end

            grad = Dgrad[octave][layer, row, col]
            hess = Dhess[octave][layer, row, col]

            update = -inv(hess) * grad
            if any(@. abs(update) < 0.5)
                converge = true
                break
            end

            x += update
        end

        layer, row, col = trunc.(Int, x)
        scale = 2^(octave - 1)

        # Contrast
        D = s.dpyr[octave]
        grad = Dgrad[octave][layer, row, col]
        contrast = D[layer, row, col] + sum(grad .* x) / 2
        low_contrast = contrast * num_layers < 0.04

        row = scale * row
        col = scale * col
        return Keypoint(; row=row, col=col,
                        low_contrast=low_contrast,
                        layer=layer, octave=octave,
                        localized=converge)
    end
end

const DISPATCH = Dict(:gpyr => generate_gaussian_pyramid,
                      :dpyr => generate_dog_pyramid,
                      :keypoints => compute_keypoints,
                      :localized_keypoints => compute_localized_keypoints,
                      :k => multiply_factor)

@lazy_context DISPATCH SIFT

function fit(s, image)
    s.base_image = generate_base_image(s, image)
    return s
end
