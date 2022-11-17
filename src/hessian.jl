using ImageFiltering
using OffsetArrays

struct Hessian{T,N}
    x::Array{T,N}
    cache::IdDict{CartesianIndex{N},SMatrix{N,N,T}}
    Hessian(image::Array{T,N}) where {T,N} = new{T,N}(image, Dict())
end

function hessian_at(x, idx::CartesianIndex{2})
    w = centered(x, idx)
    c = 2 * w[0, 0]

    # 1 dims
    dxx = w[1, 0] - c + w[-1, 0]
    dyy = w[0, 1] - c + w[0, -1]

    # 2 dims
    dxy = 0.25 * (w[1, 1] + w[-1, -1] - w[-1, 1] - w[1, -1])
    return @SMatrix [dxx dxy;
                     dxy dyy]
end

function hessian_at(x, idx::CartesianIndex{3})
    w = centered(x, idx)
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

function (h::Hessian)(idx::CartesianIndex)
    return Base.get!(() -> hessian_at(h.x, idx), h.cache, idx)
end

function (h::Hessian)(idx::Integer...)
    return h(CartesianIndex(idx))
end
