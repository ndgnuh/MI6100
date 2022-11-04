using ImageCore
using ImageTransformations
using ImageFiltering
using Base: @kwdef
using DataStructures: MutableLinkedList
using Base.Iterators
using LinearAlgebra
using Setfield
using LinearSolve
using Base.Threads: @threads

include(joinpath(@__DIR__, "imdiff.jl"))

Maybe{T} = Union{T,Nothing}
@kwdef mutable struct Keypoint
    angle = nothing
    octave = nothing
    sample = nothing
    pt = nothing
    response = nothing
    size = nothing
    orientation = nothing
    magnitude = nothing
    σ = nothing
end

@kwdef struct SIFT
    n_features::Int = 0
    n_octave_layers::Int = 3
    contrast_threshold::Float32 = 0.04
    edge_threshold::Float32 = 10
    σ::Float32 = 1.6
    border_width::Int = 5
end

@kwdef mutable struct SIFTResult
    num_octaves::Int = 0
    base_image::Maybe{Matrix} = nothing
    scales::Maybe{Vector{Float32}} = nothing
    g_pyramid::Maybe{Matrix} = nothing
    d_pyramid::Maybe{Matrix} = nothing
    keypoints::Maybe{MutableLinkedList{Keypoint}} = nothing
end

function fit(sift::SIFT, image)
    result = SIFTResult()
    result.base_image = compute_base_image(sift, image)
    result.num_octaves = compute_num_octaves(result.base_image)
    result.scales = compute_gaussian_kernels(sift)
    result.g_pyramid = compute_gaussian_pyramid(sift,
                                                result.base_image,
                                                result.scales,
                                                result.num_octaves)
    result.d_pyramid = compute_dog_pyramid(sift, result.g_pyramid)
    result.keypoints = compute_scale_space_extremas(sift, result.d_pyramid, result.scales)
    #= result.keypoints = localize_keypoints!(sift, result.d_pyramid, result.keypoints) =#
    return result
end

function gaussian_blur(image, σ)
    ksize = Int(round(round(σ * 2) / 2) * 2 - 1)
    kern = KernelFactors.gaussian((σ, σ), (ksize, ksize))
    return imfilter(image, kern, NA())
end

function compute_gaussian_kernels(sift)
    # Sigma
    n_samples = sift.n_octave_layers + 3
    σ = zeros(eltype(sift.σ), n_samples)
    σ[begin] = sift.σ

    # Compute σ values
    k = 2^(1 / sift.n_octave_layers)
    for i in 1:(n_samples - 1)
        σ_prev = k^(i - 1) * sift.σ
        σ_total = σ_prev * k
        σ[i + 1] = sqrt(σ_total^2 - σ_prev^2)
    end

    return σ
end

function compute_base_image(sift, image, init_sigma=0.5)
    dσ = sqrt(max(0.01, sift.σ * sift.σ - 4 * init_sigma * init_sigma))
    base = gaussian_blur(image, dσ)
    base = imresize(base; ratio=2, method=ImageTransformations.Linear())
    return @. gray(base)
end

function compute_num_octaves(image)
    return Int(round(log(minimum(size(image))) / log(2) - 1))
end

function compute_gaussian_pyramid(sift::SIFT, image, σ, num_octaves)
    Image = typeof(image)
    n_samples = sift.n_octave_layers + 3

    # Prepare
    pyramid = [zeros(eltype(image), size(image)) for i in 1:num_octaves, j in 1:n_samples]

    # Compute octaves
    for o_idx in 1:num_octaves, s_idx in 1:n_samples
        # First image is the base
        if o_idx == s_idx == 1
            pyramid[o_idx, s_idx] = image
            continue
        end

        # First sample of each octave is half of the original one
        if s_idx == 1
            pyramid[o_idx, s_idx] = imresize(pyramid[o_idx - 1, end - 3]; ratio=1 // 2)
            continue
        end

        # Other samples are gaussian blur of the previous
        pyramid[o_idx, s_idx] = gaussian_blur(pyramid[o_idx, s_idx - 1], σ[s_idx])
    end

    return pyramid
end

function compute_dog_pyramid(sift, gaussian_pyramid::Matrix)
    num_octaves, num_samples = size(gaussian_pyramid)
    dog_pyramid = [copy(m) for m in gaussian_pyramid[:, 1:(end - 1)]]
    @threads for o in 1:(num_octaves - 1)
        for s in 1:(num_samples - 1)
            @. dog_pyramid[o, s] = gaussian_pyramid[o, s + 1] - gaussian_pyramid[o, s]
        end
    end
    return dog_pyramid
end

function compute_scale_space_extremas(sift, dog_pyramid, σ)
    keypoints = MutableLinkedList{Keypoint}()
    num_octaves = last(size(dog_pyramid))
    border_width = sift.border_width
    gradients = map(eachrow(dog_pyramid)) do dog
        cdog = cat(dog...; dims=3)
        return G = im_gradients(cdog)
    end
    hessians = map(eachrow(dog_pyramid)) do dog
        cdog = cat(dog...; dims=3)
        return im_hessians(cdog)
    end

    # Find extremas
    for o in 1:num_octaves
        # gradient and hessians of current octave
        grads = gradients[o]
        hesses = hessians[o]

        # image size
        height, width = size(dog_pyramid[o, 1])

        # scan through triplet of samples for every octave
        for layer in 2:(sift.n_octave_layers + 1)
            prev = dog_pyramid[o, layer - 1]
            img = dog_pyramid[o, layer]
            next = dog_pyramid[o, layer + 1]
            for row in (1 + border_width):(height - border_width),
                col in (1 + border_width):(width - border_width)

                cval = img[row, col]
                is_maxima = all(begin
                                    r, c = row + i, col + j
                                    c1 = cval > prev[r, c]
                                    c2 = cval > next[r, c]
                                    c3 = (i == j == 0) || cval > img[r, c]
                                    c1 && c2 && c3
                                end
                                for i in -1:1, j in -1:1)
                is_minima = all(begin
                                    r, c = row + i, col + j
                                    c1 = cval < prev[r, c]
                                    c2 = cval < next[r, c]
                                    c3 = (i == j == 0) || cval < img[r, c]
                                    c1 && c2 && c3
                                end
                                for i in -1:1, j in -1:1)

                if !(is_maxima || is_minima)
                    continue
                end

                # Localize keypoints
                converge = false
                X = Float32[row, col, layer]
                update = Float32[0, 0, 0]
                for _ in 1:10
                    x, y, z = @. Int(round(X))

                    if (z < 2 || z > sift.n_octave_layers + 1 ||
                        x <= 1 + border_width || x >= height - border_width ||
                        y <= 1 + border_width || y >= width - border_width)
                        break
                    end

                    grad = [grad[x, y, z] for grad in grads]
                    hess = [hess[x, y, z] for hess in hesses]
                    update .= -solve(LinearProblem(hess, grad))

                    ur, uc, ul = update
                    if abs(ur) <= 0.5 && abs(uc) <= 0.5 && abs(ul) <= 0.5
                        converge = true
                        break
                    end

                    @. X = X + update
                end

                # Successful localization
                if !converge
                    continue
                end

                # Contrast
                X = X + update
                x, y, z = @. Int(round(X))
                D = dog_pyramid[o, z]
                grad = [grad[x, y, z] for grad in grads]
                contrast = abs(D[x, y] + sum(grad .* update) / 2)
                if contrast * sift.n_octave_layers < sift.contrast_threshold
                    continue
                end

                # Edge responses
                r = sift.edge_threshold
                H = [hess[x, y, z] for hess in hesses][1:2, 1:2]
                detH = det(H)
                if detH <= 0 || (tr(H)^2 / detH) >= (r + 1)^2 / r
                    continue
                end

                keypoint = Keypoint(; octave=o, sample=layer,
                                    pt=(row, col))
                keypoint.response = contrast
                keypoint.pt = (x * 2^(o - 1), y * 2^(o - 1))
                keypoint.size = let
                    #= sigma * (2 ** ((image_index + extremum_update[2]) / float32(num_intervals))) * (2 ** (octave_index + 1)) =#
                    p = (z / sift.n_octave_layers)
                    σ[o] * 2^p * 2^(o + 1)
                end

                push!(keypoints, keypoint)
            end
        end
    end

    return keypoints
end

function localize_keypoint!(sift::SIFT, grads, hesses, keypoint) end

function localize_keypoints!(sift::SIFT, dog_pyramid, keypoints; num_attempts=10)
    lkeypoints = MutableLinkedList{Keypoint}()
    border_width = sift.border_width

    GH = map(eachrow(dog_pyramid)) do dog
        cdog = cat(dog...; dims=3)
        G = im_gradients(cdog)
        H = im_hessians(cdog)
        return G, H
    end

    for keypoint in keypoints
        # Row, col, sample index
        r::Int, c::Int, s::Int = keypoint.pt[1], keypoint.pt[2], keypoint.sample
        height, width = size(dog_pyramid[keypoint.octave, keypoint.sample])
        G, H = GH[keypoint.octave]
        converge = false
        update = zeros(eltype(G[1]), 3)

        for i in 1:num_attempts
            grad = [g[r, c, s] for g in G]
            hess = [h[r, c, s] for h in H]

            prob = LinearProblem(hess, grad)
            update .= -solve(prob)
            ur, uc, us = update

            # Converge
            if abs(ur) < 0.5 && abs(uc) < 0.5 && abs(us) < 0.5
                converge = true
                break
            end

            r += round(ur)
            c += round(uc)
            s += round(us)

            if (s < 2 || s > sift.n_octave_layers + 1 ||
                r < 1 + border_width || r > height - border_width ||
                c < 1 + border_width || c > width - border_width)
                break
            end
        end

        # If not converge, skip to next kpt
        if !converge
            continue
        end

        # Contrast score
        D = dog_pyramid[keypoint.octave, s]
        ur, uc, us = update
        contrast = abs(D[r, c] + transpose([g[r, c, s] for g in G]) * update * 0.5)
        if (contrast * sift.n_octave_layers) < sift.contrast_threshold
            continue
        end

        # Edge responses
        et = sift.edge_threshold
        matH = [h[r, c, s] for h in H]
        trH = tr(matH)
        detH = det(matH)
        if (trH^2 / detH) >= (et + 1)^2 / et
            continue
        end

        keypoint.response = contrast
        keypoint.pt = (r * 2^(keypoint.octave - 1), c * 2^(keypoint.octave - 1))
        keypoint.size = let
            σ = keypoint.σ
            p = (s / sift.n_octave_layers)
            σ * 2^p * 2^(keypoint.octave)
        end
        push!(lkeypoints, keypoint)
    end
    return lkeypoints
end

function rescale_keypoints!(image, keypoints)
    height, width = size(image)
    for kpt in keypoints
        row::Int, col::Int = kpt.pt
        o = kpt.octave
        row = round(row * 2.0^(o - 2))
        col = round(col * 2.0^(o - 2))
        kpt.pt = (row, col)
    end
    return keypoints
end

function imshow(img)
    @. RGB(img)
end
