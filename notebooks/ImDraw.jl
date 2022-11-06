using ImageDraw: drawifinbounds!
using Base.Iterators
using Base.Threads: @threads

struct Circle{T,F}
    center::Tuple{T,T}
    radius::F
end

struct Line
    p1::Tuple
    p2::Tuple
end

const INTP_EPS = 1e-3
const MAX_PTS = 200

function _integer_points!(pts)
    pts_ = map(pts) do (x, y)
        return convert(Int, round(x)), convert(Int, round(y))
    end
    #= idx = unique(Iterators.map(Int ∘ round, range(1, length(pts); length=MAX_PTS))) =#
    return unique(pts_)
end

function _integer_points!(T, pts)
    pts_ = Iterators.map(pts) do (x, y)
        return convert(T, round(x)), convert(T, round(y))
    end
    #= idx = unique(Iterators.map(Int ∘ round, range(1, length(pts); length=MAX_PTS))) =#
    return unique(pts_)
end

function interpolate(c::Circle{T}) where {T}
    crow, ccol = c.center
    circ::Int = (ceil(2π * c.radius))
    r = c.radius
    pts = map(range(0, 2π; length=circ)) do t
        return (crow + r * cos(t), ccol + r * sin(t))
    end
    return _integer_points!(pts)
end

function interpolate(c::Circle{T}, thickness::Int) where {T}
    if thickness == 1
        return interpolate(c)
    else
        circles = [Circle(c.center, c.radius - i + div(thickness, 2)) for i in 1:thickness]
        return mapreduce(interpolate, union, circles)
    end
end

function interpolate(l::Line)
    x1, y1 = l.p1
    x2, y2 = l.p2
    length = trunc(Int, sqrt((x1 - x2)^2 + (y1 - y2)^2)) + 1
    pts = Iterators.map(range(0, 1; length=MAX_PTS)) do t
        x = x1 * t + (1 - t) * x2
        y = x2 * t + (1 - t) * y2
        return x, y
    end
    return _integer_points!(pts)
end

function interpolate(line::Line, thickness)
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

function imdraw!(img, d, color, thickness=1)
    @threads for (row, col) in interpolate(d, thickness)
        drawifinbounds!(img, row, col, color)
    end
    return img
end

function imdraw(img, args...)
    return imdraw!(copy(img))
end

function draw_keypoint!(img, kpt, color; thickness=1, rich=true)
    Color = eltype(img)
    row, col = kpt.pt
    radius = if rich && !isnothing(kpt.size)
        kpt.size / 2
    else
        3 # OpenCV standard
    end
    center = kpt.pt[1:2]
    imdraw!(img, Circle(center, radius), Color(color...), thickness)

    if rich && !isnothing(kpt.angle)
        radius = kpt.size / 2
        angle_rad = kpt.angle * pi / 180
        row_orient = row + radius * sin(angle_rad)
        col_orient = col + radius * cos(angle_rad)
        imdraw!(img, Line((row, col), (row_orient, col_orient)), Color(color...), thickness)
    end

    img[trunc(Int, row), trunc(Int, col)] = Color(color...)
    return img
end

function draw_keypoint2!(img, kpt, color; thickness=1, rich=true)
    Color = eltype(img)
    row, col = kpt.row, kpt.col
    radius = if rich && !isnothing(kpt.size)
        kpt.size / 2
    else
        3 # OpenCV standard
    end
    imdraw!(img, Circle((row, col), radius), Color(color...), thickness)

    if rich && !isnothing(kpt.angle)
        angle_rad = kpt.angle * pi / 180
        row_orient = row + radius * sin(angle_rad)
        col_orient = col + radius * cos(angle_rad)
        imdraw!(img, Line((row, col), (row_orient, col_orient)), Color(color...), thickness)
    end

    img[row, col] = Color(color...)
    return img
end
