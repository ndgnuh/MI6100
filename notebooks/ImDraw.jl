using ImageDraw: drawifinbounds!
using Base.Iterators

struct Circle{T}
    center::Tuple{T,T}
    radius::T
end

struct Line{T}
    p1::Tuple{T,T}
    p2::Tuple{T,T}
end

const INTP_EPS = 1e-3

function _integer_points!(T, pts)
    unique(Iterators.map(pts) do (x, y)
               return convert(T, round(x)), convert(T, round(y))
           end)
end

function interpolate(c::Circle{T}) where {T}
    crow, ccol = c.center
    r = c.radius
    pts = map(0:INTP_EPS:(2 * pi)) do t
        return (crow + r * cos(t), ccol + r * sin(t))
    end
    return _integer_points!(T, pts)
end

function interpolate(c::Circle{T}, thickness::Int) where {T}
    if thickness == 1
        return interpolate(c)
    else
        circles = [Circle(c.center, c.radius - i + div(thickness, 2)) for i in 1:thickness]
        return mapreduce(interpolate, union, circles)
    end
end

function interpolate(l::Line{T}) where {T}
    x1, y1 = l.p1
    x2, y2 = l.p2
    pts = Iterators.map(0:eps(Float16):1) do t
        x = x1 * t + (1 - t) * x2
        y = x2 * t + (1 - t) * y2
        return x, y
    end
    return _integer_points!(T, pts)
end

function interpolate(line::Line{T}, thickness) where {T}
    if thickness == 1
        return interpolate(line)
    else
        e1 = Circle(line.p1, thickness)
        e2 = Circle(line.p2, thickness)
        p1s = interpolate(e1)
        p2s = interpolate(e2)
        lines = (Line(p1, p2) for (p1, p2) in zip(p1s, p2s))
        mapreduce(interpolate, union, lines)
        #= x1, y1 = l.p1 =#
        #= x2, y2 = l.p2 =#
        #= d = div(thickness, 2) + 1 =#
        #= lines = map(1:thickness) do i =#
        #=     offset = i - d =#
        #=     return Line((x1 + offset, y1 + offset), (x2 + offset, y2 + offset)) =#
        #= end =#
        #= return mapreduce(interpolate, union, lines) =#
    end
end

function imdraw(img, d, color, thickness=1)
    img = copy(img)
    for (row, col) in interpolate(d, thickness)
        drawifinbounds!(img, row, col, color)
    end
    return img
end
