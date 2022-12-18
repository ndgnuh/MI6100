using ImageFiltering
using LinearAlgebra
using OffsetArrays
using Base.Threads

#= struct Gradient{T,N} =#
#=     x::Array{T,N} =#
#=     cache::IdDict{CartesianIndex{N},SVector{N,T}} =#
#=     magnitude::IdDict{CartesianIndex{N},T} =#
#=     angle::IdDict{CartesianIndex{N},T} =#
#=     Gradient(image::Array{T,N}) where {T,N} = new{T,N}(image, Dict(), Dict(), Dict()) =#
#= end =#

#= function gradient_at(x, idx::CartesianIndex{2}) =#
#=     w = centered(x, idx) =#
#=     dx = (w[1, 0] - w[-1, 0]) / 2 =#
#=     dy = (w[0, 1] - w[0, -1]) / 2 =#
#=     return @SVector [dx, dy] =#
#= end =#

#= function gradient_at(x, idx::CartesianIndex{3}) =#
#=     w = centered(x, idx) =#
#=     dx = (w[1, 0, 0] - w[-1, 0, 0]) / 2 =#
#=     dy = (w[0, 1, 0] - w[0, -1, 0]) / 2 =#
#=     dz = (w[0, 0, 1] - w[0, 0, -1]) / 2 =#
#=     return @SVector [dx, dy, dz] =#
#= end =#

#= function magnitude_at(g::Gradient, idx::CartesianIndex) =#
#=     Base.get!(g.magnitude, idx) do =#
#=         grad = g(idx) =#
#=         return norm(grad) =#
#=     end =#
#= end =#
#= magnitude_at(g::Gradient, coords::Integer...) = magnitude_at(g, CartesianIndex(coords)) =#

#= function angle_at(g::Gradient, idx::CartesianIndex) =#
#=     Base.get!(g.angle, idx) do =#
#=         grad = g(idx) =#
#=         return atan(last(grad) / first(grad)) =#
#=     end =#
#= end =#
#= angle_at(g::Gradient, coords::Integer...) = angle_at(g, CartesianIndex(coords)) =#

#= function (g::Gradient)(idx::CartesianIndex) =#
#=     return Base.get!(() -> gradient_at(g.x, idx), g.cache, idx) =#
#= end =#

#= function (g::Gradient)(idx::Integer...) =#
#=     return g(CartesianIndex(idx)) =#
#= end =#


# VERSION 2
#
using LoopVectorization

function imgradient(I::Array{T,3}) where {T}
    m, n, p = size(I)
    grad = zeros(T, m, n, p, 3)
    @turbo for i = 2:m-1, j = 2:n-1, k = 2:p-1
        grad[i, j, k, 1] = (I[i+1, j, k] - I[i-1, j, k]) / 2
        grad[i, j, k, 2] = (I[i, j+1, k] - I[i, j-1, k]) / 2
        grad[i, j, k, 3] = (I[i, j, k+1] - I[i, j, k-1]) / 2
    end
    return grad
end

function imgradient(I::Array{T,2})::Array{T,3} where {T}
    m, n = size(I)
    grad = zeros(T, m, n, 2)
    @turbo for i = 2:m-1, j = 2:n-1
        grad[i, j, 1] = (I[i+1, j] - I[i-1, j]) / 2
        grad[i, j, 2] = (I[i, j+1] - I[i, j-1]) / 2
    end
    return grad
end

function imhessian(I::Array{T,3}) where {T}
    m, n, p = size(I)
    hess = zeros(T, m, n, p, 3, 3)
    @turbo for i = 2:m-1, j = 2:n-1, k = 2:p-1
        hess[i, j, k, 1, 1] = I[i+1, j, k] + I[i-1, j, k] - I[i, j, k] * 2
        hess[i, j, k, 2, 2] = I[i, j+1, k] + I[i, j-1, k] - I[i, j, k] * 2
        hess[i, j, k, 3, 3] = I[i, j, k+1] + I[i, j, k-1] - I[i, j, k] * 2
        hess[i, j, k, 1, 2] = (I[i+1, j+1, k] + I[i-1, j-1, k] - I[i+1, j-1, k] - I[i-1, j+1, k]) / 4
        hess[i, j, k, 1, 3] = (I[i+1, j, k+1] + I[i-1, j, k-1] - I[i+1, j, k-1] - I[i-1, j, k+1]) / 4
        hess[i, j, k, 2, 3] = (I[i, j+1, k+1] + I[i, j-1, k-1] - I[i, j+1, k-1] - I[i, j-1, k+1]) / 4
    end
    @turbo for i = 2:m-1, j = 2:n-1, k = 2:p-1
        hess[i, j, k, 2, 1] = hess[i, j, k, 1, 2]
        hess[i, j, k, 3, 1] = hess[i, j, k, 1, 3]
        hess[i, j, k, 3, 2] = hess[i, j, k, 2, 3]
    end
    return hess
end

function imhessian(I::Array{T,2}) where {T}
    m, n = size(I)
    hess = zeros(T, m, n, 2, 2)
    @turbo for i = 2:m-1, j = 2:n-1
        hess[i, j, 1, 1] = I[i+1, j] + I[i-1, j] - I[i, j] * 2
        hess[i, j, 2, 2] = I[i, j+1] + I[i, j-1] - I[i, j] * 2
        hess[i, j, 3, 3] = I[i, j+1] + I[i, j-1] - I[i, j] * 2
        hess[i, j, 1, 2] = (I[i+1, j+1] + I[i-1, j-1] - I[i+1, j-1] - I[i-1, j+1]) / 4
    end
    @turbo for i = 2:m-1, j = 2:n-1
        hess[i, j, 2, 1] = hess[i, j, 1, 2]
    end
    return hess
end
