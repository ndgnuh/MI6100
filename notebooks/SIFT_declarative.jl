using ImageFiltering
using Memoize
using Images
using StackViews
using Base: @kwdef
using Setfield: @set, @set!
using LinearAlgebra
using LinearSolve
using Chain
using StatsBase

module ImDiff
include(joinpath(@__DIR__, "imdiff.jl"))
end

struct ScaleSpace
    image::Matrix{Float32}
    num_layers::Int
    σ::Float32
end

@kwdef struct Keypoint
    angle = nothing
    octave = nothing
    row = nothing
    col = nothing
    layer = nothing
    response = nothing
    size = nothing
    orientation = nothing
    magnitude = nothing
end

function _gradients_at_center_3d(w)
    w = centered(w)
    dx = (w[1, 0, 0] - w[-1, 0, 0]) / 2
    dy = (w[0, 1, 0] - w[0, -1, 0]) / 2
    dz = (w[0, 0, 1] - w[0, 0, -1]) / 2
    return [dx, dy, dz]
end

function _hessians_at_center_3d(w)
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

function contrast_threshold(s)
    return 0.04
end

function stack(arrs, dims)
    return StackView(arrs...; dims=dims)
end

@memoize function get_octave_size(s, octave)
    return size(compute_dog_image(s, octave, 1))
end

@memoize function get_blur(s, layer::Int)
    @assert layer >= 1
    k = get_multiply_factor(s)
    if layer == 1
        return s.σ
    else
        σp = k^(layer - 2) * s.σ
        σk = k * σp
        return sqrt(σk^2 - σp^2)
    end
end

@memoize function get_num_layers(s)
    return s.num_layers
end
@memoize function get_num_samples(s)
    return s.num_layers + 3
end

function get_blurs(s)
    return [get_blur(s, i) for i in 1:get_num_samples(s)]
end

function gaussian_blur(image, σ)
    ksize = Int(round(round(σ * 2) / 2) * 2 - 1)
    kern = KernelFactors.gaussian((σ, σ), (ksize, ksize))
    return imfilter(image, kern, NA())
end

@memoize function get_multiply_factor(s)
    return 2.0f0^(1 / get_num_layers(s))
end

#= @memoize function compute_base_image(s, init_sigma=0.5) =#
#=     image = s.image =#
#=     σ = get_blur(s, 1) =#
#=     base = imresize(image; ratio=2, method=ImageTransformations.Linear()) =#
#=     dσ = sqrt(max(0.01, σ * σ - 4 * init_sigma * init_sigma)) =#
#=     base = gaussian_blur(base, dσ) =#
#=     return @. gray(base) =#
#= end =#

@memoize function compute_base_image(sift, init_sigma=0.5)
    σ = get_blur(sift, 1)
    dσ = sqrt(max(0.01, σ * σ - 4 * init_sigma * init_sigma))
    base = sift.image
    base = imresize(base; ratio=2, method=ImageTransformations.Linear())
    base = imfilter(base, Kernel.gaussian(dσ))
    return @. gray(base)
end

@memoize function compute_gaussian_image(s, octave, layer)::Matrix{Float32}
    @assert octave >= 1 && layer >= 1
    if octave == layer == 1
        return compute_base_image(s)
    end

    if layer > 1
        kern = Kernel.gaussian(get_blur(s, layer))
        prev_image = compute_gaussian_image(s, octave, layer - 1)
        # Layer = 2
        # dσ = sqrt((σk)^2 - σ^2)
        # L(x, y, kσ) = L(x, y, σ) * G(x, y, dσ)
        return imfilter(prev_image, kern)
    end

    if layer == 1
        octave_base = compute_gaussian_image(s, octave - 1, get_num_layers(s))
        return imresize(octave_base; ratio=1 // 2)
    end
end

@memoize function compute_dog_image(s, octave, layer)::Matrix{Float32}
    D = compute_gaussian_image(s, octave, layer)
    Dnext = compute_gaussian_image(s, octave, layer + 1)
    return (Dnext - D)
end

@memoize function num_octaves(s)
    return trunc(Int, round(log(minimum(size(compute_base_image(s)))) / log(2) - 1))
end

@memoize function compute_gaussian_pyramid(s)
    image = compute_base_image(s)
    return [compute_gaussian_image(s, octave, layer)
            for octave in 1:num_octaves(s), layer in 1:get_num_samples(s)]
end

@memoize function compute_dog_pyramid(s)
    return [compute_dog_image(s, octave, layer)
            for octave in 1:num_octaves(s), layer in 1:(get_num_samples(s) - 1)]
end

@memoize function compute_dog_pyramid(s, octave::Int)
    return [compute_dog_image(s, octave, layer) for layer in 1:(get_num_samples(s) - 1)]
end

@memoize function compute_scale_space_extrema(s; no_localize=false)
    threshold = floor(0.5 * contrast_threshold(s) / 3 * 255)
    candidates = @chain begin
        Iterators.map(1:num_octaves(s)) do octave
            dg = stack(compute_dog_pyramid(s, octave), 3)
            height, width, num_layers = size(dg)
            extremas = mapwindow(dg, (3, 3, 3);
                                 indices=(2:height, 2:width, 2:(num_layers - 1))) do cube
                center = cube[2, 2, 2]
                return all(center .>= cube) || all(center .<= cube)
            end
            if no_localize
                return map(findall(extremas)) do idx
                    row, col, layer = idx.I
                    return Keypoint(; octave=octave,
                                    row=row * 2^(octave - 1),
                                    col=col * 2^(octave - 1),
                                    layer=layer)
                end
            end

            kpts = @chain begin
                findall(extremas)
                Iterators.map(_) do idx
                    return _localize_keypoints(s, octave, idx)
                end
                Iterators.filter(!isnothing, _)
                Iterators.map(kpt -> compute_keypoint_orientation(s, kpt), _)
                Iterators.filter(!isnothing, _)
            end
            return Set(kpts)
        end
        Iterators.filter(!isempty, _)
        reduce(union, _; init=Set())
    end

    return candidates
end

@memoize function compute_dog_gradients(s)
    return map(1:num_octaves(s)) do octave
        dog = stack(compute_dog_pyramid(s, octave), 3)
        grads = ImDiff.im_gradients(dog)
        height, width = get_octave_size(s, octave)
        indices = CartesianIndices((height, width, get_num_samples(s) - 1))
        return [H[idx] for idx in indices, H in grads]
    end
end

@memoize function compute_dog_hessians(s)
    return map(1:num_octaves(s)) do octave
        dog = stack(compute_dog_pyramid(s, octave), 3)
        hessians = ImDiff.im_hessians(dog)
        height, width = get_octave_size(s, octave)
        indices = CartesianIndices((height, width, get_num_samples(s) - 1))
        return [H[idx] for idx in indices, H in hessians]
    end
end

function _localize_keypoints(s, octave, idx)
    row, col, layer = idx.I
    x = Float32[row, col, layer]
    update = zeros(Float32, 3)
    converge = false
    num_attemps = 30
    height, width = get_octave_size(s, octave)
    num_samples = get_num_samples(s)

    # Localize keypoint
    for i in 1:num_attemps
        row, col, layer = trunc.(Int, x)
        if (row < 2 || row > height - 1 ||
            col < 2 || col > width - 1 ||
            layer < 2 || layer > num_samples - 1)
            return nothing
        end

        Dp = compute_dog_image(s, octave, layer - 1)
        D = compute_dog_image(s, octave, layer)
        Dn = compute_dog_image(s, octave, layer + 1)
        cube = cat(Dp[(row - 1):(row + 1), (col - 1):(col + 1)],
                   D[(row - 1):(row + 1), (col - 1):(col + 1)],
                   Dn[(row - 1):(row + 1), (col - 1):(col + 1)]; dims=3)

        #= grad = compute_dog_gradients(s)[octave][row, col, layer, :] =#
        grad = _gradients_at_center_3d(cube)
        hess = _hessians_at_center_3d(cube)
        #= hess = compute_dog_hessians(s)[octave][row, col, layer, :, :] =#

        if det(hess) == 0
            return nothing
        end

        update .= -solve(LinearProblem(hess, grad))

        if any(@. abs(update) < 0.5)
            converge = true
            break
        end

        x = x + update
    end

    # Localization fail
    if !converge
        return nothing
    end

    # Final grad and hess
    Dp = compute_dog_image(s, octave, layer - 1)
    D = compute_dog_image(s, octave, layer)
    Dn = compute_dog_image(s, octave, layer + 1)
    cube = cat(Dp[(row - 1):(row + 1), (col - 1):(col + 1)],
               D[(row - 1):(row + 1), (col - 1):(col + 1)],
               Dn[(row - 1):(row + 1), (col - 1):(col + 1)]; dims=3)

    #= grad = compute_dog_gradients(s)[octave][row, col, layer, :] =#
    grad = _gradients_at_center_3d(cube)
    hess = _hessians_at_center_3d(cube)

    # Compute contrast
    row, col, layer = trunc.(Int, x)
    D = compute_dog_image(s, octave, layer)
    #= grad = compute_dog_gradients(s)[octave][row, col, layer, :] =#
    contrast = D[row, col] + sum(grad .* update) / 2
    if abs(contrast * get_num_layers(s)) < contrast_threshold(s)
        return nothing
    end

    # Compute edge response
    # Remove keypoints on the edge since they are not rotate invariants
    r = 10.0
    H = hess[1:2, 1:2]
    detH = det(H)
    trH = tr(H)
    if detH <= 0 || trH^2 * r >= (r + 1)^2 * detH
        return nothing
    end

    σ = get_blur(s, octave)
    size = σ * 2^(layer / get_num_layers(s)) * 2^octave
    scale = 2^(octave - 1)
    row = row * scale
    col = col * scale
    kpt = Keypoint(; size=size,
                   octave=octave,
                   row=row,
                   col=col,
                   layer=layer)
    return kpt
end

function compute_gaussian_gradients(s, octave, layer)
    L = compute_gaussian_image(s, kpt.octave, kpt.layer)
    return stack(ImDiff.im_gradients(L), 3)
end

function compute_angle_image(s, octave, layer)
    L = compute_gaussian_image(s, octave, layer)
    Grow, Gcol = ImDiff.im_gradients(L)
    angle = @. atan(Grow / Gcol)
    return angle
end

function compute_keypoints_with_orientation(s, kpt)
    scale_factor = 1.5
    # compare with keypoint.size computation in localizeExtremumViaQuadraticFit()
    return scale = scale_factor * kpt.size / (2.0^(kpt.octave + 1))
end

function compute_keypoint_orientation(s, kpt)
    angle = compute_angle_image(s, kpt.octave, kpt.layer)
    indices = get_indices_around(s, kpt)

    # Compute histogram
    edges = range(0, 360; length=36)
    angles = [angle[row, col] for (row, col) in indices]
    hist = fit(Histogram, angles, edges)
    @info hist
    #= nbins = 10 =#
    #= hist = Dict{Int,Int}() =#
    #= for (row, col) in indices =#
    #=     dcol = L[row, col + 1] - L[row, col - 1] =#
    #=     drow = L[row + 1, col] - L[row - 1, col] =#
    #=     angle = atan(dcol / drow) =#
    #=     deg = trunc(Int, angle * 360 / pi) =#
    #=     key = trunc(Int, deg / 10) =#
    #=     hist[key] = 1 + get(hist, key, 0) =#
    #= end =#

    # Principle angle
    #= max_count, idx = findmax(hist) =#
    #= @set! kpt.angle = idx * 10 =#
    return kpt
end

function get_indices_around(s, kpt)
    octave, row, col, layer = kpt.octave, kpt.row, kpt.col, kpt.layer
    height, width = get_octave_size(s, octave)
    offset = trunc(Int, kpt.size) + 1
    iter = @chain begin
        Iterators.product((-offset):offset, (-offset):offset)
        Iterators.filter(_) do args
            drow, dcol = args
            return drow^2 + dcol^2 <= kpt.size^2
        end
        Iterators.map(_) do args
            (drow, dcol) = args
            return (kpt.row + drow, kpt.col + dcol)
        end
        Iterators.filter(_) do args
            row, col = args
            return row > 1 && col > 1 && row < height - 1 && col < width - 1
        end
    end
    return Set(iter)
end
