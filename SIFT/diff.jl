using Setfield
using Base.Iterators: product
using OffsetArrays
using ImageFiltering
using StaticArrays

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
