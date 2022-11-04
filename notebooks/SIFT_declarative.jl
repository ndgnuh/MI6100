using ImageFiltering
using Memoize
using Images

struct ScaleSpace
    image::Matrix{Float32}
    num_layers::Int
    σ::Float32
end

@memoize function get_blur(s, layer::Int)
    k = 2^(1 / s.num_layers)
    if layer == 1
        return s.σ
    else
        σp = get_blur(s, layer - 1)
        σk = k * σp
        return sqrt(σk^2 - σp^2)
    end
end

@memoize function compute_gaussian_image(s, octave, layer)::Matrix{Float32}
    @assert octave >= 1 && layer >= 1
    if octave == layer == 1
        return s.image
    elseif layer == 1
        octave_base = compute_gaussian_image(s, octave - 1, s.num_layers + 3)
        return imresize(octave_base; ratio=1 // 2)
    else
        kern = Kernel.gaussian(get_blur(s, layer))
        prev_image = compute_gaussian_image(s, octave, layer - 1)
        return imfilter(prev_image, kern)
    end
end

@memoize function compute_dog_image(s, octave, layer)::Matrix{Float32}
    D = compute_gaussian_image(s, octave, layer)
    Dnext = compute_gaussian_image(s, octave, layer + 1)
    return (Dnext - D)
end

@memoize function num_octaves(s)
    return trunc(Int, round(log(minimum(size(s.image))) / log(2) - 1))
end

@memoize function compute_gaussian_pyramid(s)
    [begin
         compute_gaussian_image(s, octave, layer)
     end
     for octave in 1:num_octaves(s), layer in 1:num_samples(s)]
end

@memoize function compute_dog_pyramid(s)
    [begin
         compute_dog_image(s, octave, layer)
     end
     for octave in 1:num_octaves(s), layer in 1:num_samples(s)]
end

@memoize function num_layers(s)
    return s.num_layers
end
@memoize function num_samples(s)
    return s.num_layers + 3
end

@memoize function get_octave_size(s, octave)
    return size(compute_dog_image(s, octave, 1))
end

@memoize function compute_scale_space_extrema(s)
    dpyr = compute_gaussian_pyramid(s)
    candidates = mapreduce(union, 1:num_octaves(s)) do octave
        height, width = get_octave_size(s, octave)
        all_indices = Iterators.product(2:(height - 1),
                                        2:(width - 1),
                                        2:(num_layers(s) - 1))
        extremas = Iterators.filter(all_indices) do (row, col, layer)
            is_point_extrema(s, octave, row, col, layer)
        end

        Set(extremas)
    end
end

@memoize function is_point_extrema(s, octave, row, col, layer)
    curr = compute_dog_image(s, octave, layer)
    threshold = 0.03
    center_value = curr[row, col]

    iter = Iterators.product(-1:1, -1:1, -1:1)

    function compare_function(cmp::Function)
        return function (drow, dcol, dlayer)
            if drow == dcol == dlayer == 0
                return true
            end
            compare_dog = compute_dog_image(s, octave, layer + dlayer)
            compare_value = compare_dog[row + drow, col + dcol]
            cmp(center_value, compare_value)
        end
    end
    is_minima = Iterators.map(compare_function(<), iter)
    is_maxima = Iterators.map(compare_function(>), iter)
    all(is_minima) || all(is_maxima)
end
