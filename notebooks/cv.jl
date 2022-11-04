module OpenCV
using OpenCV_jll
using CxxWrap
include(joinpath(OpenCV_jll.artifact_dir, "OpenCV", "src", "OpenCV.jl"))
end
