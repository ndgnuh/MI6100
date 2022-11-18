using Base: @kwdef

@kwdef mutable struct SIFTMethod
    sigma = 1.6
    num_layers = 3
    assumed_blur = 0.5
end

@kwdef struct Keypoint{T<:AbstractFloat}
    scale::Int
    row::Int
    col::Int
    orientation::T
    magnitude::T
end

function isinframe(coord::NTuple{N,T}, shape::NTuple{N,T}, border=zeros(T, N)) where {N,T}
    return all(@. (coord > border) && (coord <= shape - border))
end
