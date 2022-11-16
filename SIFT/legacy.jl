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

struct DiffCtx{T,N}
    image::Array{T,N}
    grad_cache::Dict{CartesianIndex{N},SVector{N,T}}
    hess_cache::Dict{CartesianIndex{N},SMatrix{N,N,T}}

    DiffCtx(image::Array{T,N}) where {T,N} = new{T,N}(image, Dict(), Dict())
end

function compute_gradients_at_center_3d(w, idx)
    w = centered(w, idx)
    dx = (w[1, 0, 0] - w[-1, 0, 0]) / 2
    dy = (w[0, 1, 0] - w[0, -1, 0]) / 2
    dz = (w[0, 0, 1] - w[0, 0, -1]) / 2
    return @SVector [dx, dy, dz]
end

function compute_hessians_at_center_3d(w, idx)
    w = centered(w, idx)
    c = 2 * w[0, 0, 0]

    # 1 dims
    dxx = w[1, 0, 0] - c + w[-1, 0, 0]
    dyy = w[0, 1, 0] - c + w[0, -1, 0]
    dzz = w[0, 0, 1] - c + w[0, 0, -1]

    # 2 dims
    dxy = 0.25 * (w[1, 1, 0] + w[-1, -1, 0] - w[-1, 1, 0] - w[1, -1, 0])
    dxz = 0.25 * (w[1, 0, 1] + w[-1, 0, -1] - w[-1, 0, 1] - w[1, 0, -1])
    dyz = 0.25 * (w[0, 1, 1] + w[0, -1, -1] - w[0, -1, 1] - w[0, 1, -1])
    return @SMatrix [dxx dxy dxz;
                     dxy dyy dyz;
                     dxz dyz dzz]
end

function im_gradients(ctx::DiffCtx, idx::CartesianIndex)
    if !haskey(ctx.grad_cache, idx)
        ctx.grad_cache[idx] = compute_gradients_at_center_3d(ctx.image, idx)
    end
    return ctx.grad_cache[idx]
end

function im_hessians(ctx::DiffCtx, idx::CartesianIndex)
    if !haskey(ctx.hess_cache, idx)
        ctx.hess_cache[idx] = compute_hessians_at_center_3d(ctx.image, idx)
    end
    return ctx.hess_cache[idx]
end

function offset(idx, dim, by)
    for (d, b) in zip(dim, by)
        idx = @set idx[d] = idx[d] + b
    end
    return idx
end

function compute_diff_kernel(coefs, n, dims)
    kernel = zeros(fill(n, dims)...)
    idx = CartesianIndex(fill(div(n, 2) + 1, dims)...)
    for (dim, dt, c) in coefs
        kernel[offset(idx, dim, dt)] += c
    end
    return kernel
end

function compute_hessian_kernels(dims)
    coefs = map(product(1:dims, 1:dims)) do (d1, d2)
        if d1 != d2
            c1 = ([d1, d2], [1, 1], 1 / 4)
            c2 = ([d1, d2], [1, -1], -1 / 4)
            c3 = ([d1, d2], [-1, 1], -1 / 4)
            c4 = ([d1, d2], [-1, -1], 1 / 4)
            (c1, c2, c3, c4)
        else
            c1 = (d1, 1, 1)
            c2 = (d1, 0, -2)
            c3 = (d1, -1, 1)
            (c1, c2, c3)
        end
    end
    return compute_diff_kernel.(coefs, 3, dims)
end

function compute_gradient_kernels(dims)
    coefs = map(1:dims) do d
        c1 = (d, 1, 1 / 2)
        c2 = (d, -1, -1 / 2)
        return (c1, c2)
    end
    return compute_diff_kernel.(coefs, 3, dims)
end

function im_gradients(img)
    kernels = compute_gradient_kernels(ndims(img))
    @. imfilter((img,), kernels)
end

function im_hessians(img)
    kernels = compute_hessian_kernels(ndims(img))
    @. imfilter((img,), kernels)
end

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

