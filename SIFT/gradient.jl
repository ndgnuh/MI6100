using ImageFiltering
using OffsetArrays

struct Gradient{T, N}
    x::Array{T, N}
    cache::IdDict{CartesianIndex{N}, SVector{N, T}}
    Gradient(image::Array{T, N}) where {T, N} = new{T, N}(image, Dict())
end

function (g::Gradient)(idx::CartesianIndex{2})
    w = centered(g.x, cp=idx)
    dx = (w[1, 0] - w[-1, 0]) / 2
    dy = (w[0, 1] - w[0, -1]) / 2
    return @SVector [dx, dy]
end

function (g::Gradient)(idx::CartesianIndex{3})
    w = centered(g.x, idx)
    dx = (w[1, 0, 0] - w[-1, 0, 0]) / 2
    dy = (w[0, 1, 0] - w[0, -1, 0]) / 2
    dz = (w[0, 0, 1] - w[0, 0, -1]) / 2
    return @SVector [dx, dy, dz]
end

function Base.get!(g::Gradient, idx::CartesianIndex)
    Base.get!(() -> g(idx), g.cache, idx)
end
