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
const REFERENCE_PATCH_WIDTH_SCALAR = 6 * REFERENCE_LOCALITY

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
