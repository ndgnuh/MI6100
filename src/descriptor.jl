using UnPack
using StaticArrays

const DESCRIPTOR_LOCALITY = 0

function compute_descriptor(keypoint, L, octave_index)
    magnitudes, orientations = let
        dr = shift(L, 0, 1, 0) - shift(L, 0, -1, 0)
        dc = shift(L, 0, 0, 1) - shift(L, 0, 0, -1)
        mag = @. sqrt(dr^2 + dc^2)
        ori = @. atan(dc, dr) % (2 * pi)
        mag, ori
    end

    desc = (@SVector zeros(Float32, 128))
    @unpack row, col, scale = keypoint
    dist = get_octave_pixel_distance(octave_index)
    
    #= max_width = @chain begin =#
    #=     (np.sqrt(2) * const.descriptor_locality * sigma) / pixel_dist =#
    #=     trunct(Int, _) =#
    #= end =#
end
