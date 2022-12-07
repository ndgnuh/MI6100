using UnPack
using StaticArrays
using .Constants

function compute_descriptor(keypoint, L, octave_index)
    # Magnitudes and orientations
    M, ϕ = calculate_gradient_stats(L)
    desc = (@SVector zeros(Float32, 128))
    @unpack row, col, scale = keypoint

    # Stats
    pixel_dist = get_octave_pixel_distance(octave_index)
    max_width = @chain begin
        sqrt(2.0f0) * Constants.DESCRIPTOR_LOCALITY * sigma / pixel_dist
        trunct(Int, _)
    end

    if !isinframe((scale, row, col), size(L))
        return descriptor
    end

    ϕ_patch = ϕ[scale, row - max_width: row + max_width, col - max_width: col + max_width]
    M_patch = M[scale, row - max_width: row + max_width, col - max_width: col + max_width]
    patch_size = size(M_patch)
    center_offset = [coord[1] - y, coord[2] - x]
    rel_patch_coords = relative_patch_coordinates(center_offset, patch_shape, pixel_dist, sigma,
                                                  keypoint.orientation)
    magnitude_patch = mask_outliers(magnitude_patch, rel_patch_coords, const.descriptor_locality)
    orientation_patch = (orientation_patch - keypoint.orientation) % (2 * np.pi)
    weights = weighting_matrix(center_offset, patch_shape, octave_idx, sigma, const.descriptor_locality)
    magnitude_patch = magnitude_patch * weights
    coords_rel_to_hists = rel_patch_coords[None] - histogram_centers[..., None, None]
    hists_magnitude_patch = mask_outliers(magnitude_patch[None], coords_rel_to_hists, const.inter_hist_dist, 1)
    hists_magnitude_patch = interpolate_2d_grid_contribution(hists_magnitude_patch, coords_rel_to_hists)
    hists = interpolate_1d_hist_contribution(hists_magnitude_patch, orientation_patch).ravel()
    keypoint.descriptor = normalize_sift_feature(hists)
    described_keypoints.append(keypoint)
end
