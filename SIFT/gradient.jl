using ImageFiltering
using LinearAlgebra
using OffsetArrays

struct Gradient{T,N}
    x::Array{T,N}
    cache::IdDict{CartesianIndex{N},SVector{N,T}}
    magnitude::IdDict{CartesianIndex{N},T}
    angle::IdDict{CartesianIndex{N},T}
    Gradient(image::Array{T,N}) where {T,N} = new{T,N}(image, Dict(), Dict(), Dict())
end

function gradient_at(x, idx::CartesianIndex{2})
    w = centered(x, idx)
    dx = (w[1, 0] - w[-1, 0]) / 2
    dy = (w[0, 1] - w[0, -1]) / 2
    return @SVector [dx, dy]
end

function gradient_at(x, idx::CartesianIndex{3})
    w = centered(x, idx)
    dx = (w[1, 0, 0] - w[-1, 0, 0]) / 2
    dy = (w[0, 1, 0] - w[0, -1, 0]) / 2
    dz = (w[0, 0, 1] - w[0, 0, -1]) / 2
    return @SVector [dx, dy, dz]
end

function magnitude_at(g::Gradient, idx::CartesianIndex)
    Base.get!(g.magnitude, idx) do
        grad = g(idx)
        return norm(grad)
    end
end
magnitude_at(g::Gradient, coords::Integer...) = magnitude_at(g, CartesianIndex(coords))

function angle_at(g::Gradient, idx::CartesianIndex)
    Base.get!(g.angle, idx) do
        grad = g(idx)
        return atan(last(grad) / first(grad))
    end
end
angle_at(g::Gradient, coords::Integer...) = angle_at(g, CartesianIndex(coords))

function (g::Gradient)(idx::CartesianIndex)
    return Base.get!(() -> gradient_at(g.x, idx), g.cache, idx)
end

function (g::Gradient)(idx::Integer...)
    return g(CartesianIndex(idx))
end
