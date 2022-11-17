using ImageFiltering
using ImageTransformations

function compute_num_octaves(sz)
    return trunc(Int, round(log(minimum(sz)) / log(2) - 1))
end

function generate_base_image(s, image)
    dsigma = sqrt(s.sigma^2 - 4 * s.assumed_blur^2)
    base = imresize(image; ratio=2)
    base = imfilter(base, Kernel.gaussian(dsigma))
    return base
end

function get_blur(s, layer::Int)
    @assert layer >= 1
    k = 2.0^(1 / s.num_layers)
    if layer == 1
        return s.sigma
    else
        σp = k^(layer - 2) * s.sigma
        σk = k * σp
        return sqrt(σk^2 - σp^2)
    end
end

function get_blurs(s)
    return [get_blur(s, i) for i in 1:get_num_samples(s)]
end

function generate_gaussian_pyramid(s)
    base_image = s.base_image
    T = eltype(base_image)
    base_height, base_width = size(base_image)
    num_octaves = compute_num_octaves((base_height, base_width))
    num_samples = s.num_layers + 3

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
        k = s.k
        sigma = k^(layer) * s.sigma
        gpyr[octave][layer, :, :] .= imfilter(base, Kernel.gaussian(sigma))
    end
    return gpyr
end

function generate_dog_pyramid(s)
    gpyr = s.gpyr
    dpyr = map(gpyr) do Ls
        return diff(Ls; dims=1)
    end
    return dpyr
end

