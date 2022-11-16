module Draw

using ImageDraw: drawifinbounds!
using ImageDraw

include(joinpath(@__DIR__, "struct.jl"))

function circle!(image, x0, y0, rad, color; thickness=1)
    rad2 = rad^2
    for d1 in (-rad):rad
        d2 = trunc(Int, sqrt(rad2 - d1^2))
        d1 = trunc(Int, d1)
        ImageDraw.drawifinbounds!(image, CartesianIndex(x0 + d1, y0 + d2), color)
        ImageDraw.drawifinbounds!(image, CartesianIndex(x0 + d1, y0 - d2), color)
        ImageDraw.drawifinbounds!(image, CartesianIndex(x0 + d2, y0 + d1), color)
        ImageDraw.drawifinbounds!(image, CartesianIndex(x0 - d2, y0 + d1), color)
    end
    if thickness > 1
        delta = trunc(Int, thickness / 2)
        drange = ((-delta + (thickness + 1) % 2):delta)

        for d in drange
            circle!(image, x0, y0, rad + d, color; thickness=1)
        end
    end
    return image
end

#= function draw_keypoint!(image, keypoint, color) =#
#=     radius = something(keypoint.size, 3) =#
#=     circle!(image, keypoint.row, keypoint.col, radius, color) =#
#=     ImageDraw.draw!(image, ImageDraw.Point(keypoint.row, keypoint.col), color) =#
#=     if !isnothing(keypoint.angle) && !isnothing(keypoint.size) =#
#=         row = keypoint.row =#
#=         col = keypoint.col =#
#=         angle = keypoint.angle =#
#=         size = keypoint.size =#
#=         orr = trunc(Int, row + size * cos(angle + π / 2)) =#
#=         orc = trunc(Int, col + size * sin(angle + π / 2)) =#
#=         ImageDraw.bresenham(image, row, col, orr, orc, color) =#
#=     end =#
#=     return image =#
#= end =#

function draw_keypoint!(image, keypoint, color)
    @assert isinframe((keypoint.row, keypoint.col), size(image))
    radius = something(keypoint.magnitude, 3)
    circle!(image, keypoint.row, keypoint.col, radius, color)
    ImageDraw.draw!(image, ImageDraw.Point(keypoint.col, keypoint.row), color)
    if !isnothing(keypoint.orientation) && !isnothing(keypoint.magnitude)
        row = keypoint.row
        col = keypoint.col
        orientation = keypoint.orientation
        magnitude = keypoint.magnitude
        orr = trunc(Int, row + magnitude * cos(orientation + π / 2))
        orc = trunc(Int, col + magnitude * sin(orientation + π / 2))
        ImageDraw.bresenham(image, row, col, orr, orc, color)
    end
    return image
end

function draw_keypoints!(image, keypoints, color)
    foreach(kpt -> draw_keypoint!(image, kpt, color), keypoints)
    return image
end

end
