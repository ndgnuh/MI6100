using Parameters

@with_kw mutable struct SIFT
    sigma = 1.6
    num_layers = 3
    assumed_blur = 0.5
end

@with_kw struct Keypoint{T<:AbstractFloat}
    scale::Int
    row::Int
    col::Int
    orientation::T
    magnitude::T
end

function isinframe(coord::NTuple{N,T}, shape::NTuple{N,T}) where {N,T}
    return all(@. (coord > 2) && (coord < shape - 1))
end
