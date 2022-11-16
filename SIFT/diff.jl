using ImageFiltering

function shift(array::Array{T,3}, s::Int, y::Int, x::Int) where {T}
    """ Takes a 3D tensor and shifts it in a specified direction.
    Args:
        array: The 3D array that is to be shifted.
        shift_spec: The shift specification for each of
            the 3 axes. E.g., [1, 0, 0] will make the
            element (x,x,x) equal element (x+1, x+1, x+1) in
            the original image, effectively shifting the
            image "to the left", along the first axis.
    Returns:
        shifted: The shifted array.
    """
    padded = padarray(array, Pad(:replicate, 1, 1, 1))
    shifted = padded[(begin + 1 + s):(end - 1 + s),
                     (begin + 1 + y):(end - 1 + y),
                     (begin + 1 + x):(end - 1 + x)]
    return shifted
end

function shift(array::Array{T,2}, y::Int, x::Int) where {T}
    padded = padarray(array, Pad(:replicate, 1, 1))
    shifted = padded[(begin + 1 + y):(end - 1 + y),
                     (begin + 1 + x):(end - 1 + x)]
    return shifted
end

function shift(array::Array{T,1}, x::Int) where {T}
    padded = padarray(array, Pad(:replicate, 1, 1))
    shifted = padded[(begin + 1 + x):(end - 1 + x)]
    return shifted
end

function derivatives(octave)
    o = octave

    # Gradient
    ds = (shift(o, 1, 0, 0) - shift(o, -1, 0, 0)) / 2
    dr = (shift(o, 0, 1, 0) - shift(o, 0, -1, 0)) / 2
    dc = (shift(o, 0, 0, 1) - shift(o, 1, 0, -1)) / 2

    # Hessian
    dss = (shift(o, 1, 0, 0) + shift(o, -1, 0, 0) - 2 * o)
    drr = (shift(o, 0, 1, 0) + shift(o, 0, -1, 0) - 2 * o)
    dcc = (shift(o, 0, 0, 1) + shift(o, 0, 0, -1) - 2 * o)
    dsr = (shift(o, 1, 1, 0) - shift(o, 1, -1, 0) - shift(o, -1, 1, 0) +
           shift(o, -1, -1, 0)) / 4
    dsc = (shift(o, 1, 0, 1) - shift(o, 1, 0, -1) - shift(o, -1, 0, 1) +
           shift(o, -1, 0, -1)) / 4
    drc = (shift(o, 0, 1, 1) - shift(o, 0, 1, -1) - shift(o, 0, -1, 1) +
           shift(o, 0, -1, -1)) / 4

    grad = cat(ds, dr, dc; dims=4)
    hess = cat(cat(dss, dsr, dsc; dims=4),
               cat(dsr, drr, drc; dims=4),
               cat(dsc, drc, dcc; dims=4);
               dims=5)
    return grad, hess
end
