using Accessors
using Memoize
using Base: IdDict
using Parameters
using UnPack

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

include(joinpath(@__DIR__, "struct.jl"))
include(joinpath(@__DIR__, "gaussian.jl"))
include(joinpath(@__DIR__, "visualize.jl"))

function multiply_factor(s)
    return 2.0f0^(1 / s.num_layers)
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

function compute_gradients_at_center_3d(w)
    w = centered(w)
    dx = (w[1, 0, 0] - w[-1, 0, 0]) / 2
    dy = (w[0, 1, 0] - w[0, -1, 0]) / 2
    dz = (w[0, 0, 1] - w[0, 0, -1]) / 2
    return [dx, dy, dz]
end

function compute_hessians_at_center_3d(w)
    w = centered(w)
    c = 2 * w[0, 0, 0]

    # 1 dims
    dxx = w[1, 0, 0] - c + w[-1, 0, 0]
    dyy = w[0, 1, 0] - c + w[0, -1, 0]
    dzz = w[0, 0, 1] - c + w[0, 0, -1]

    # 2 dims
    dxy = 0.25 * (w[1, 1, 0] + w[-1, -1, 0] - w[-1, 1, 0] - w[1, -1, 0])
    dxz = 0.25 * (w[1, 0, 1] + w[-1, 0, -1] - w[-1, 0, 1] - w[1, 0, -1])
    dyz = 0.25 * (w[0, 1, 1] + w[0, -1, -1] - w[0, -1, 1] - w[0, 1, -1])
    return [dxx dxy dxz;
            dxy dyy dyz;
            dxz dyz dzz]
end

include(joinpath(@__DIR__, "diff.jl"))
function compute_localized_keypoints(s::SIFT)
    Dgrad = map(s.dpyr) do dog
        return im_gradients(dog)
    end
    @info "Gradient done"
    Dhess = map(s.dpyr) do dog
        return mapwindow(compute_hessians_at_center_3d, dog, (3, 3, 3))
    end
    @info "Hessian done"
    return
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
