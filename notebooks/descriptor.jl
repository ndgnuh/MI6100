### A Pluto.jl notebook ###
# v0.19.16

using Markdown
using InteractiveUtils

# ╔═╡ 8e0eac2b-dadc-479b-8613-9b3783c9bbcc
module Constants
"""
This file contains the SIFT algorithm's parameters.
When a docstring mentions 'the standard configuration of SIFT',
it refers to these parameters, as provided in the repository's main branch.
These parameters were chosen in accordance with the AOS paper.

Mentions of 'AOS' refer to
    Anatomy of the SIFT Method. Otero, Ives Rey. Diss. École normale supérieure de Cachan-ENS Cachan, 2015.
Mentions of 'Lowe' refer to
    Sift-the scale invariant feature transform. Lowe. Int. J, 2(91-110), 2, 2004.
"""

const EPS = eps(Float32)


###################
# Octaves
###################

# The number of Gaussian and Difference of Gaussian octaves that are generated,
# Referred to as n_oct in AOS
const NR_OCTAVES = 8

# The number of scales generated per Gaussian octave.
# Referred to as n_spo in AOS
const SCALES_PER_OCTAVE = 3

# Auxiliary scales per octave. Required to generate enough scales in the Gaussian
# scale space such that the Difference of Gaussian layers span the desired scales.
# Here, desired scales means the doubling of the Gaussian blur's standard deviation
# after each DoG octave and having 3 DoG layers per octave in which extrema can be found.
# See AOS Figure 2.(b). and Figure 5.
const AUXILIARY_SCALES = 3

# The assumed geometric distance between pixels in the original image.
# Referred to as delta_in in AOS.
const ORIG_PIXEL_DIST = 1

# The pixel distance in the first layer of the first octave,
# i.e., upscaled version of the input image.
# Referred to as delta_min in AOS.
const MIN_PIXEL_DIST = 0.5f0

# The amount by which the input image is upscaled for the first octave.
const FIRST_UPSCALE = ORIG_PIXEL_DIST / MIN_PIXEL_DIST

# The assumed blur level of the input image
# Referred to as sigma_in in AOS.
const ORIG_SIGMA = 0.5f0

# The desired blur level for the first octave's first layer.
# Referred to as sigma_min in AOS.
const MIN_SIGMA = 0.8f0

# The sigma required to move from the input image to the first octave's first layer.
# See AOS section 2.2 formula 6.
const INIT_SIGMA = 1.0f0 / MIN_PIXEL_DIST * sqrt(MIN_SIGMA^2 - ORIG_SIGMA^2)


#########################
# Keypoint Tests
#########################

# Determines whether a Difference of Gaussian extrema is large enough.
# Referred to as C_dog in AOS. See AOS section 3.3.
const MAGNITUDE_THRESH = 0.015f0 * ((2^(1.0f0 / SCALES_PER_OCTAVE) - 1) / (2^(1.0f0 / 3) - 1))

# Before attempting to interpolate a DoG extrema's value, it must have
# a magnitude larger than this threshold.
const COARSE_MAGNITUDE_THRESH = 0.85f0 * MAGNITUDE_THRESH

# Determines whether a coordinate offset of the interpolated extremum
# is too large. If it is too large, the interpolated offset must be recalculated relative
# to a different sampling point. This threshold is et to 0.5 in Lowe and 0.6 in AOS.
# See AOS section 3.2.
const OFFSET_THRESH = 0.6f0

# Determines whether the the eigenvalue ratio of the hessian is too large.
# A eigenvalue ratio larger than this value means the keypoint must be discarded,
# as it probably lies on an edge or other poorly defined feature.
const EDGE_RATIO_THRESH = 10

# Maximum number of attempts for interpolating an extrema
# See AOS section 3.4 algorithm 6.
const MAX_INTERPOLATIONS = 3


#########################
# Reference orientation
#########################

# The number of bins in the reference orientation histogram.
# Referred to as n_bins in AOS. See AOS section 4.1.
const NR_BINS = 36

# Controls how "local" or "close" to the keypoint the reference orientation
# analysis is performed. For example, this value is used to set the size of the
# patch used for reference orientation finding, and for weighting the contribution
# of gradients in the local neighborhood to find this reference orientation.
# Increasing this value would result in a larger neighborhood being considered.
# Referred to as lambda_ori in AOS.
const REFERENCE_LOCALITY = 1.5f0

# An additional constant to control the size of the reference orientation patch.
# See AOS Figure 7.
const REFERENCE_PATCH_WIDTH_SCALAR = trunc(Int, 6 * REFERENCE_LOCALITY)

# Number of smoothing steps performed with a three-tap box filter ([1, 1, 1])
# on the reference orientation histogram. See AOS section 4.1.B.
const NR_SMOOTH_ITER = 6

# The magnitude (relative to the histogram's largest peak)
# that an orientation bin must reach to be considered a reference
# orientation. Referred to as 't' in AOS. See blue line in AOS Figure 8.
const REL_PEAK_THRESH = 0.8f0

# The number of bins to the left and right of a peak that are ignored when searching
# for the next potential local maximum in the reference orientation histogram.
const MASK_NEIGHBORS = 4

# The maximum number of local maxima that will be found in a gradient orientation histogram.
const MAX_ORIENTATIONS_PER_KEYPOINT = 2


#########################
# Descriptor
#########################

# Controls how "local" or "close" to the keypoint the descriptor analysis
# is performed. For example, this value is used to set the size of the
# patch used for descriptor finding, and for weighting the contribution
# of gradients in the descriptor. Increasing this value would result in
# a larger neighborhood being considered. Referred to as lambda_descr in AOS.
const DESCRIPTOR_LOCALITY = 6

# The number of rows in the square descriptor grid. Each cell in this grid
# has a histogram associated with it. Referred to as n_hist in AOS.
const NR_DESCRIPTOR_HISTOGRAMS = 4

# The distance between histogram centers along the x or y axis.
const INTER_HIST_DIST = 1.0f0 * DESCRIPTOR_LOCALITY / NR_DESCRIPTOR_HISTOGRAMS

# The number of orientation bins in a descriptor histogram.
# Referred to as n_ori in AOS.
const NR_DESCRIPTOR_BINS = 8

# The width of an descriptor histogram's orientation bin.
const DESCRIPTOR_BIN_WIDTH = NR_DESCRIPTOR_BINS / Float32(2 * pi)

# The normalized maximum amount of mass that may be assigned to a single
# bin in the SIFT feature, i.e., the concatenated descriptor histograms.
# See AOS section 4.2. The SIFT feature vector.
const DESCRIPTOR_CLIP_MAX = 0.2f0


#########################
# Matching
#########################

# The distance factor between first and second nearest neighbor for accepting a feature match.
# I.e, create a match if:  first_nn_dist < second_nn_dist * `rel_dist_match_thresh`
const REL_DIST_MATCH_THRESH = 0.6f0

end


# ╔═╡ 00af14ac-7643-11ed-37e0-b34e14a6739c
function hist_center()
	xs = []
	ys = []
	bin_width = 2 * Constants.DESCRIPTOR_LOCALITY / Constants.NR_DESCRIPTOR_HISTOGRAMS

	hist_center_offset = bin_width / 2
	start_coord = -Constants.DESCRIPTOR_LOCALITY + hist_center_offset

	for row_idx in 1:Constants.NR_DESCRIPTOR_HISTOGRAMS
		for col_idx in 1:Constants.NR_DESCRIPTOR_HISTOGRAMS
			y = start_coord + bin_width * row_idx
			x = start_coord + bin_width * col_idx
			push!(ys, y)
			push!(xs, x)
		end
	end
	centers = [xs ys]
end

# ╔═╡ 1c0ca0c7-4f8a-496e-af88-6a465ceb883f
hist_center()

# ╔═╡ 7cbe22b2-efa6-4ce7-b84e-31f012ee54ff
""" 
	relative_patch_coordinate(
		center_offset,
		patch_shape,
		pixel_dist,
		sigma,
		keypoint_orientation
	)

Calculates the coordinates of pixels in a descriptor patch,
	relative to the keypoint. Keypoints have an orientation and
	therefore introduce an oriented x and y axis. This is why
	the relative coordinates are the result of a rotation.
	See Lowe section 5 and AOS section 4.2.

## Args:
- center_offset: The keypoint's offset from the patch's center.
- patch_shape: The shape of a descriptor patch including padding.
- pixel_dist: The distance between adjacent pixels.
- sigma: The scale of layer where the keypoint was found.
- keypoint_orientation: The orientation of the keypoint in radians.

## Returns:
- rel_coords: The y & x coordinates of pixels in a descriptor patch
		relative to the keypoint's location and orientation.
"""
function relative_patch_coordinate(
	center_offset,
	patch_shape,
	pixel_dist,
	sigma,
	keypoint_orientation
)
	y_len, x_len = patch_shape
	center = @. patch_shape / 2 + center_offset
	y_ids = 1:y_len
	x_ids = 1:x_len

	# coordinates are rotated to align with keypoint's orientation
	rel_xs = (
		(xs - center[2]) * cos(keypoint_orientation) +
		(ys - center[1]) * sin(keypoint_orientation)
	) / (sigma / pixel_dist)
	rel_ys = (
		(centers[2] - xs) * sin(keypoint_orientation) +
		(ys - centers[1]) * cos(keypoint_orientation)
	) / (sigma / pixel_dist)

	rel_coords = [rel_xs rel_ys]
end

# ╔═╡ 20aa7fd8-775f-42c5-84a8-5473ec81f786
""" 
```
mask_outliers(magnitude_patch, rel_patch_coords, threshold, dims)
```

Masks outliers in a patch. Here, an outlier has a distance from the patch's center keypoint along the y or x axis that is larger than the threshold.

## Args:
- `magnitude_patch`: The gradient magnitudes in the patch.
- `rel_patch_coords`: The y & x coordinates of pixels in a descriptor patch relative to the keypoint's location and potentially orientation.
- `threshold`: Distance in y and x after which a point is masked to 0.
- `axis`: The axis along which the max between y & x is found.
## Returns:
- `magnitude_patch`: The  gradient magnitudes in the patch after masking.
"""
function mask_outliers(
	magnitude_patch,
	rel_patch_coords,
	threshold,
	dims,
)
	mask = max(abs(rel_patch_coords), dims=axis) <= threshold
	return magnitude_patch * mask
end

# ╔═╡ fa9eabc5-bb1d-4e11-8353-ef366db881ed
""" 
```julia
interpolate_2d_grid_contibution(magnitude_path, coords_rel_to_hist, descriptor_locality=$(Constants.DESCRIPTOR_LOCALITY))
```

Interpolates gradient contributions to surrounding histograms. In other words: Calculates to what extent gradients in a descriptor patch contribute to a histogram, based on the gradient's pixel's y & x distance to that histogram's location. See AOS section 4.2 and figure 10 and Lowe section 6. This function performs the interpolation for all histograms at once via broadcasting.

## Args:
- `magnitude_path`: The gradient magnitudes in a descriptor patch, used to weigh gradient contributions. For the standard configuration, this array is of shape (2, 32, 32) with semantics (y_or_x, patch_row, patch_col).
- `coords_rel_to_hist`: The coordinates of pixels in a descriptor patch, relative to a histograms location. For the standard configuration, this array is of shape (2, 16, 32, 32) after axes swap, with semantics (y_or_x, hist_idx, patch_row, patch_col).

## Returns:
- `magnitude_path`: The gradient magnitudes in a descriptor patch after interpolating their contributions for each histogram. For the standard configuration, this array is of shape (16, 32, 32), with semantics (hist_idx, patch_row, patch_col).
"""
function interpolate_2d_grid_contibution(
	magnitude_path,
	coords_rel_to_hist,
	descriptor_locality=Constants.DESCRIPTOR_LOCALITY
)
	coords_rel_to_hist = permutedims(coords_rel_to_hist, (1, 2))
	xs, ys = abs.(coords_rel_to_hist)
	y_contrib = 1 - (ys / (0.5 * descriptor_locality))
	x_contrib = 1 - (xs / (0.5 * descriptor_locality))
	contrib = y_contrib * x_contrib
	return magnitude_path * contrib
end

# ╔═╡ f2934e94-5aab-486d-8a97-e869cdd59cfe
function interpolate_1d_hist_contribution(
	magnitude_path,
	orientation_patch
)
	nr_hists = size(magnitude_path)[1]
	orientation_patch = r
end

# ╔═╡ df273fac-5a41-4a10-a1c0-83c3f889b163
x = rand(3, 5)

# ╔═╡ 6ee39ebb-f501-41e6-9204-027fef604bd5
repeat(reshape(x, (1, 3, 5)), 4)

# ╔═╡ 6d6e974c-3432-4e2f-9bd6-091e5163d989


# ╔═╡ Cell order:
# ╟─8e0eac2b-dadc-479b-8613-9b3783c9bbcc
# ╠═00af14ac-7643-11ed-37e0-b34e14a6739c
# ╠═1c0ca0c7-4f8a-496e-af88-6a465ceb883f
# ╟─7cbe22b2-efa6-4ce7-b84e-31f012ee54ff
# ╟─20aa7fd8-775f-42c5-84a8-5473ec81f786
# ╟─fa9eabc5-bb1d-4e11-8353-ef366db881ed
# ╠═f2934e94-5aab-486d-8a97-e869cdd59cfe
# ╠═df273fac-5a41-4a10-a1c0-83c3f889b163
# ╠═6ee39ebb-f501-41e6-9204-027fef604bd5
# ╠═6d6e974c-3432-4e2f-9bd6-091e5163d989
