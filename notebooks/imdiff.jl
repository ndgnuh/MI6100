using Setfield
using Base.Iterators: product
using OffsetArrays
using ImageFiltering

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
    kernel
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
            c3 = (d1, -1, -1)
            (c1, c2, c3)
        end
    end
    compute_diff_kernel.(coefs, 3, dims)
end

function compute_gradient_kernels(dims)
    coefs = map(1:dims) do d
        c1 = (d, 1, 1 / 2)
        c2 = (d, -1, -1 / 2)
        (c1, c2)
    end
    compute_diff_kernel.(coefs, 3, dims)
end

function im_gradients(img)
    kernels = compute_gradient_kernels(ndims(img))
    @. imfilter((img,), kernels)
end

function im_hessians(img)
    kernels = compute_hessian_kernels(ndims(img))
    @. imfilter((img,), kernels)
end
