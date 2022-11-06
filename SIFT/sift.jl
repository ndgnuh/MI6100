using Accessors
using Memoize
using Base: IdDict
using Parameters

"""
Unpack object properties
```
obj = (x = 1, y = 2)
@unpack y, x = obj

print(y, x) # == (2, 1)
```
"""
macro unpack(asn)
    lhs_ = first(asn.args)
    lhs::Vector{Symbol} = if lhs_ isa Symbol
        [lhs_]
    else
        lhs_.args
    end
    rhs = last(asn.args)
    asg = map(lhs) do key
        return Expr(:(=), key, getproperty(eval(rhs), key))
    end
    return Expr(:toplevel, asg..., Expr(:tuple, lhs...))
end

macro lazy_context(dispatch_e, T)
    dispatch = eval(dispatch_e)
    quote
        function Base.getproperty(ctx::$(esc(T)), prop::Symbol)
            value = getfield(ctx, prop)
            if isnothing(value) && prop in keys($(esc(dispatch_e)))
                setproperty!(ctx, prop, $(esc(dispatch_e))[prop](ctx))
            end
            return getfield(ctx, prop)
        end
    end
end

include(joinpath(@__DIR__, "struct.jl"))
include(joinpath(@__DIR__, "gaussian.jl"))
include(joinpath(@__DIR__, "visualize.jl"))

function multiply_factor(s)
    return 2.0f0^(1 / s.num_layers)
end

## THIS MUST BE RUN AT THE END OF THE MODULE

Maybe{T} = Union{Nothing,T}

@with_kw mutable struct SIFT
    sigma = 1.6
    num_layers = 3
    assumed_blur = 0.5
    base_image = nothing
    k = nothing
    sigmas = nothing
    gpyr = nothing
    dpyr = nothing
    keypoints = nothing
end
const DISPATCH = Dict(:gpyr => generate_gaussian_pyramid,
                      :dpyr => generate_dog_pyramid,
                      :keypoints => compute_keypoints,
                      :k => multiply_factor)

@lazy_context DISPATCH SIFT

function fit(s, image)
    s.base_image = generate_base_image(s, image)
    return s
end
