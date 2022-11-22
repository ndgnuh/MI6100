using DataStructures

function smooth_histogram(X, kernel=SVector(1 // 3, 1 // 3, 1 // 3); iterations=6)
    for _ in 1:iterations
        X .= imfilter(X, kernel)
    end
    return X
end

function calculate_gradient_stats(L)
    dy = (shift(L, 0, 1, 0) - shift(L, 0, -1, 0)) / 2
    dx = (shift(L, 0, 0, 1) - shift(L, 0, 0, -1)) / 2

    # Modulo is to shift range from [-pi, pi] to [0, 2pi]
    magnitudes = @. sqrt(dy^2 + dx^2)
    orientations = @. atan(dy, dx) % (2 * pi)

    return magnitudes, orientations
end


#= function assign_orientations(gaussian_octave, octave_index, coords, nbins::Int=36; init_σ::F, num_scales, k::F) where {F} =#
#=     r, φ = calculate_gradient_stats(gaussian_octave) =#
#=     shape = size(gaussian_octave) =#

#=     # σ values for each octaves =#
#=     absolute_σ = compute_absolute_σ(init_σ, num_scales, k) =#

#=     # TODO: calculate radius =#
#=     radius = maximum(Iterators.flatten(trunc.(Int, absolute_σ[end] * 1.5 * magnitudes))) =#
#=     #= radius = maximum(Iterators.flatten(radiuses)) =# =#
#=     orientation_bins = padarray((@. trunc(Int, nbins / 2 / pi * orientations)), =#
#=         Pad(:reflect, radius, radius, radius)) =#
#=     return mapreduce(union, coords; init=NamedTuple[]) do coord =#
#=         scale, row, col = coord.I =#
#=         σ = absolute_σ[scale+1] =#
#=         radius = trunc(Int, σ * 1.5 * magnitudes[scale, row, col]) =#
#=         ori = orientation_bins[scale, (row-radius):(row+radius), =#
#=             (col-radius):(col+radius)] =#
#=         mag = orientation_bins[scale, (row-radius):(row+radius), =#
#=             (col-radius):(col+radius)] =#
#=         hist = fit(Histogram, view(ori, :); nbins=nbins).weights =#
#=         smooth = imfilter(hist, (@SVector [1 // 3, 1 // 3, 1 // 3])) =#
#=         dom_weight, dom_orientation = findmax(smooth) =#
#=         dom_orientations = findall(smooth) do weight =#
#=             return abs((dom_weight - weight) / dom_weight) <= 0.8 =#
#=         end =#
#=         #= dom_orientations = [dom_orientation] =# =#

#=         Iterators.map(dom_orientations) do orientation =#
#=             size_ = init_σ * (2^(scale / num_scales)) * 2.0f0^(octave_index - 2) =#
#=             #= ratio = get_octave_pixel_distance(octave_index) =# =#
#=             return Keypoint{Float32}(; scale=scale, =#
#=                 octave=octave_index, =#
#=                 row=row, # trunc(Int, row * 2.0f0^(octave_index - 2)), =#
#=                 col=col, #trunc(Int, col * 2.0f0^(octave_index - 2)), =#
#=                 magnitude=trunc(Int, σ), =#
#=                 orientation=deg2rad(nbins * orientation)) =#
#=         end =#
#=     end =#
#= end =#

"""
    compute_weighting_matrix(center_offset, patch_shape, octave_idx, σ, locality=Constants.REFERENCE_LOCALITY)

Calculates a Gaussian weighting matrix.
This matrix determines the weight that gradients
in a keypoint's neighborhood have when contributing
to the keypoint's orientation histogram. See AOS section 4,
Lowe section 5.

## Parameters
- `center_offset`: The keypoint's offset from the patch's center.
- `patch_shape`: The shape of the patch. The generated weighting
    matrix will need to have the same shape to allow weighting
    by multiplication.
- `ω`: The index of the octave.
- `σ`: The scale of the Difference of Gaussian layer where
    the keypoint was found.
- `locality`: The locality of the weighting. A higher locality
    is associated with a larger neighborhood of gradients.
    See lambda parameters in AOS section 6 table 4.
"""
function compute_weighting_matrix(
    patch_shape, ω, σ;
    locality=Constants.REFERENCE_LOCALITY
)
    pixel_dist = get_octave_pixel_distance(ω)
    y_len, x_len = patch_shape
    x_center = trunc(Int, x_len / 2)
    y_center = trunc(Int, y_len / 2)
    xs, ys = @chain begin
        Iterators.product(1:x_len, 1:y_len)
        collect(_)
        invert(_)
    end
    rel_dists = @. sqrt((xs - x_center)^2 + (ys - y_center)^2)
    abs_dists = @. rel_dists * pixel_dist
    denom = @. 2 * ((locality * σ)^2)
    weights = @. exp(-((abs_dists^2) / denom))
    return weights
end

function compute_patch_size(octave_index, σ)
    pixel_dist = get_octave_pixel_distance(octave_index)
    return (Constants.REFERENCE_PATCH_WIDTH_SCALAR * σ) / pixel_dist
end

function assign_orientations(gaussian_octave, octave_index, coords, nbins::Int=36; init_σ::F, num_scales, k::F) where {F}
    # magnitude and angle
    M, Φ = calculate_gradient_stats(gaussian_octave)
    shape = size(gaussian_octave)
    edges = range(0, 2 * π, length=nbins + 1)

    # σ values for each octaves
    absolute_σ = compute_absolute_σ(init_σ, num_scales, k)
    keypoints = MutableLinkedList{Keypoint}()
    for coord in coords
        scale, row, col = coord.I
        σ = absolute_σ[scale]
        patch_size = compute_patch_size(octave_index, σ)
        patch_radius = trunc(Int, patch_size / 2)

        # Check if the coord has a valid region around it
        valid = isinframe((scale, row, col), shape, patch_radius)
        if !valid
            continue
        end

        # patch
        Φ_patch = @view M[scale,
            row-patch_radius:row+patch_radius,
            col-patch_radius:col+patch_radius]

        contributions = @chain begin
            @view M[scale,
                row-patch_radius:row+patch_radius,
                col-patch_radius:col+patch_radius]
            (_ * compute_weighting_matrix(size(_), octave_index, σ))
            weights(_)
        end
        hist = fit(
            Histogram,
            view(Φ_patch, :),
            contributions,
            edges
        )
        hist_weights = smooth_histogram(hist.weights)

        dominate_ϕ = find_histogram_peaks(hist_weights)

        for ϕ in dominate_ϕ
            keypoint = Keypoint(
                row=row,
                col=col,
                scale=scale,
                octave=octave_index,
                magnitude=4.0f0,
                orientation=ϕ
                #= coord=coord, octave_idx=octave_idx, orientation=orientation =#
            )
            push!(keypoints, keypoint)
        end
        #= contribution = map(*, weight, M_patch) =#
    end

    return keypoints
end

"""
    find_histogram_peaks(hist::AbstractVector)

Return the orientations of the peaks in radians.
In other words, the dominant orientations of gradients
in the local neighborhood of the keypoint.

Finds peaks in the gradient orientations histogram,
and returns the corresponding orientations in radians.
Peaks are the maximum bin and bins that lie within 0.80
of the mass of the maximum bin. See AOS section 4.1 and
Lowe section 5. When the modulo operator is used in this
function, it is to  account for the fact that the first
and last bin are neighbors, namely, the rotations by 0
and 2pi radians.

## Arguments
- `hist`: Histogram where each bin represents an orientation, in other 
words, an angle of a gradient. The mass of the bin is determined
by the number of gradients in the keypoint's local neighborhood
that have that orientation.
"""
function find_histogram_peaks(hist)
    nbins = length(hist)
    orientations = MutableLinkedList{Float32}()
    hist_masked = copy(hist)

    # Find first max and mask the histogram
    global_max, max_idx = findmax(hist)
    left = hist[mod(max_idx - 1 - 1, nbins)+1]
    right = hist[mod(max_idx + 1 - 1, nbins)+1]
    orientation::Float32 = @chain begin
        (2.0f0 * pi * max_idx) / nbins + (pi / nbins) * ((left - right) / (left - 2 * global_max + right))
        _ % (2 * pi)
    end
    for j in 1:(Constants.MASK_NEIGHBORS+1)
        hist_masked[mod(max_idx - j - 1, nbins)+1] = 0
        hist_masked[mod(max_idx + j - 1, nbins)+1] = 0
    end
    push!(orientations, orientation)

    # Other peaks
    for i in 1:(Constants.MAX_ORIENTATIONS_PER_KEYPOINT-1)
        new_max, max_idx = findmax(hist_masked)
        if new_max < global_max * 0.8
            break
        end

        left = hist[mod(max_idx - 1 - 1, nbins)+1]
        right = hist[mod(max_idx + 1 - 1, nbins)+1]
        orientation = @chain begin
            (2.0f0 * pi * max_idx) / nbins + (pi / nbins) * ((left - right) / (left - 2 * global_max + right))
            _ % (2 * pi)
        end
        for j in 1:(Constants.MASK_NEIGHBORS+1)
            hist_masked[mod(max_idx - j - 1, nbins)+1] = 0
            hist_masked[mod(max_idx + j - 1, nbins)+1] = 0
        end
        push!(orientations, orientation)
    end

    return orientations
end
