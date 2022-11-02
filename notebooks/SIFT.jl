using ImageCore
using ImageTransformations
using ImageFiltering
using Base: @kwdef
using DataStructures: MutableLinkedList
using Base.Iterators
using LinearAlgebra
using Setfield
using LinearSolve

@kwdef mutable struct Keypoint
    angle = nothing
    octave = nothing
    sample = nothing
    pt = nothing
    response = nothing
    size = nothing
    orientation = nothing
    magnitude = nothing
    σ = nothing
end

@kwdef struct SIFT
    n_features::Int = 0
    n_octave_layers::Int = 3
    contrast_threshold::Float32 = 0.04
    edge_threshold::Float32 = 10
    σ::Float32 = 1.6
end

function offset(idx, dim, by)
    for (d, b) in zip(dim, by)
        idx = @set idx[d] = idx[d] + b
    end
    return idx
end

function compute_hessians_at(arr, idx::CartesianIndex)
    n = length(idx)
    map(product(1:n, 1:n)) do (d1, d2)
        i1 = offset(idx, (d1, d2), (1, 1))
        i2 = offset(idx, (d1, d2), (1, -1))
        i3 = offset(idx, (d1, d2), (-1, 1))
        i4 = offset(idx, (d1, d2), (-1, -1))
        return (arr[i1] - arr[i2] - arr[i3] + arr[i4]) / 4
    end
end

function compute_gradients_at(arr, idx::CartesianIndex)
    n = length(idx)
    map(1:n) do d
        inext = offset(idx, d, 1)
        iprev = offset(idx, d, -1)
        return (arr[inext] - arr[iprev]) / 2
    end
end

function imhessians(img, kernel, border="replicate")
    grads = imgradients(img, kernel, border)
    hessians = [imgradients(grad, kernel, border) for
                grad in grads]
    n = length(grads)
    return [hessians[i][j] for i in 1:n, j in 1:n]
end

function get_num_octaves(image)
    return convert(Int, floor(log(minimum(size(image))) / log(2) - 1))
end

function preprare_input(image)
    return @. convert(Float32, (image))
    #= return @. convert(Gray{Float32}, (image)) =#
end

function compute_gaussian_pyramid(sift, image, n_octaves=get_num_octaves(image))
    image = preprare_input(image)
    Image = typeof(image)
    n_samples = sift.n_octave_layers + 3

    # Sigma
    σ = zeros(Float32, n_samples)
    σ[begin] = sift.σ

    # Prepare
    pyramid = [zeros(eltype(image), size(image)) for i in 1:n_octaves, j in 1:n_samples]

    # Compute σ values
    k = 2^(1 / sift.n_octave_layers)
    for i in 1:(n_samples - 1)
        σ_prev = k^(i - 1) * sift.σ
        σ_total = σ_prev * k
        σ[i + 1] = sqrt(σ_total^2 - σ_prev^2)
    end

    # Compute octaves
    for o_idx in 1:n_octaves, s_idx in 1:n_samples
        # First image is the base
        if o_idx == s_idx == 1
            pyramid[o_idx, s_idx] = image
            continue
        end

        # First sample of each octave is half of the original one
        if s_idx == 1
            pyramid[o_idx, s_idx] = imresize(pyramid[o_idx - 1, s_idx]; ratio=1 // 2)
            continue
        end

        # Other samples are gaussian blur of the first
        kern = Kernel.gaussian(σ[s_idx])
        pyramid[o_idx, s_idx] = imfilter(pyramid[o_idx, s_idx - 1], kern)
    end

    return pyramid
end

function compute_dog_pyramid(sift, gaussian_pyramid::Matrix)
    n_octaves = get_num_octaves(gaussian_pyramid[1])
    dog_pyramid = [zeros(eltype(m), size(m)) for m in gaussian_pyramid[:, 1:(end - 1)]]
    for o in 1:(n_octaves - 1), s in 1:(sift.n_octave_layers + 2)
        @. dog_pyramid[o, s] = gaussian_pyramid[o, s + 1] - gaussian_pyramid[o, s]
    end
    return dog_pyramid
end

function compute_scale_space_extremas(sift, dog_pyramid, border_width=5)
    keypoints = MutableLinkedList{Keypoint}()
    n_octaves = get_num_octaves(dog_pyramid[1])

    # Sigma
    σ = zeros(Float32, sift.n_octave_layers + 3)
    σ[begin] = sift.σ
    k = 2^(1 / sift.n_octave_layers)
    for i in 1:(sift.n_octave_layers + 3 - 1)
        σ_prev = k^(i - 1) * sift.σ
        σ_total = σ_prev * k
        σ[i + 1] = sqrt(σ_total^2 - σ_prev^2)
    end

    for o in 1:n_octaves
        height, width = size(dog_pyramid[o, 1])
        for i in 2:(sift.n_octave_layers + 1)
            prev = dog_pyramid[o, i - 1]
            img = dog_pyramid[o, i]
            next = dog_pyramid[o, i + 1]
            for row in (1 + border_width):(height - border_width),
                col in (1 + border_width):(width - border_width)

                cval = img[row, col]
                is_maxima = all(begin
                                    r, c = row + i, col + j
                                    c1 = cval > prev[r, c]
                                    c2 = cval > next[r, c]
                                    c3 = (i == j == 0) || cval > img[r, c]
                                    c1 && c2 && c3
                                end
                                for i in -1:1, j in -1:1)
                is_minima = all(begin
                                    r, c = row + i, col + j
                                    c1 = cval < prev[r, c]
                                    c2 = cval < next[r, c]
                                    c3 = (i == j == 0) || cval < img[r, c]
                                    c1 && c2 && c3
                                end
                                for i in -1:1, j in -1:1)

                if is_maxima || is_minima
                    push!(keypoints, Keypoint(; σ=σ[o], octave=o, sample=i, pt=(row, col)))
                end
            end
        end
    end

    return keypoints
end
function compute_grad_hess(image, kernel, border="replicate")
    dims = ndims(image)
    grads = imgradients(image, kernel, border)
    hesses = [imgradients(grad, kernel, border) for grad in grads]

    G = cat(grads...; dims=dims + 1)
    H = cat([cat(hesses_...; dims=dims + 2) for hesses_ in hesses]...; dims=dims + 1)
    return G, H
end
function localize_keypoints(sift::SIFT, dog_pyramid, keypoints; num_attempts=10,
                            border_width=5)
    lkeypoints = MutableLinkedList{Keypoint}()
    GH = map(eachrow(dog_pyramid)) do dog
        cdog = cat(dog...; dims=3)
        return compute_grad_hess(cdog, KernelFactors.prewitt)
    end

    for keypoint in keypoints
        # Row, col, sample index
        r::Int, c::Int, s::Int = keypoint.pt[1], keypoint.pt[2], keypoint.sample
        height, width = size(dog_pyramid[keypoint.octave, keypoint.sample])
        converge = false
        G, H = GH[keypoint.octave]

        for i in 1:num_attempts
            grad = G[r, c, s, :]
            hess = H[r, c, s, :, :]

            prob = LinearProblem(hess, grad)
            ur, uc, us = -solve(prob)

            # Converge
            if abs(ur) < 0.5 && abs(uc) < 0.5 && abs(us) < 0.5
                r += round(ur)
                c += round(uc)
                s += round(us)
                converge = true
                break
            end

            r += round(ur)
            c += round(uc)
            s += round(us)

            if (s < 2 || s > sift.n_octave_layers + 1 ||
                r < 1 + border_width || r > height - border_width ||
                c < 1 + border_width || c > width - border_width)
                break
            end
        end

        # If not converge, skip to next kpt
        if !converge
            continue
        end

        # Contrast score
        D = dog_pyramid[keypoint.octave, s]
        contrast = D[r, c] + sum(G[r, c, s, :] .* [r, c, s]) / 2
        if contrast * sift.n_octave_layers < sift.contrast_threshold
            continue
        end

        # Edge responses
        #= r = 10 =#
        #= H = H[r, c, s, :, :] =#
        #= trH = tr(H) =#
        #= detH = det(H) =#
        #= if (trH^2 / detH) >= (r + 1)^2 / r =#
        #=     continue =#
        #= end =#

        keypoint.response = contrast
        keypoint.pt = (r * 2^(keypoint.octave - 1), c * 2^(keypoint.octave - 1))
        keypoint.size = let
            σ = keypoint.σ
            p = (s / sift.n_octave_layers)
            σ * 2^p * 2^(keypoint.octave)
        end
        push!(lkeypoints, keypoint)
    end
    return lkeypoints
end

function imshow(img)
    @. Gray(img)
end

# {
#     CV_TRACE_FUNCTION();
#
#     const float img_scale = 1.f/(255*SIFT_FIXPT_SCALE);
#     const float deriv_scale = img_scale*0.5f;
#     const float second_deriv_scale = img_scale;
#     const float cross_deriv_scale = img_scale*0.25f;
#
#     float xi=0, xr=0, xc=0, contr=0;
#     int i = 0;
#
#     for( ; i < SIFT_MAX_INTERP_STEPS; i++ )
#     {
#         int idx = octv*(nOctaveLayers+2) + layer;
#         const Mat& img = dog_pyr[idx];
#         const Mat& prev = dog_pyr[idx-1];
#         const Mat& next = dog_pyr[idx+1];
#
#         Vec3f dD((img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1))*deriv_scale,
#                  (img.at<sift_wt>(r+1, c) - img.at<sift_wt>(r-1, c))*deriv_scale,
#                  (next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c))*deriv_scale);
#
#         float v2 = (float)img.at<sift_wt>(r, c)*2;
#         float dxx = (img.at<sift_wt>(r, c+1) + img.at<sift_wt>(r, c-1) - v2)*second_deriv_scale;
#         float dyy = (img.at<sift_wt>(r+1, c) + img.at<sift_wt>(r-1, c) - v2)*second_deriv_scale;
#         float dss = (next.at<sift_wt>(r, c) + prev.at<sift_wt>(r, c) - v2)*second_deriv_scale;
#         float dxy = (img.at<sift_wt>(r+1, c+1) - img.at<sift_wt>(r+1, c-1) -
#                      img.at<sift_wt>(r-1, c+1) + img.at<sift_wt>(r-1, c-1))*cross_deriv_scale;
#         float dxs = (next.at<sift_wt>(r, c+1) - next.at<sift_wt>(r, c-1) -
#                      prev.at<sift_wt>(r, c+1) + prev.at<sift_wt>(r, c-1))*cross_deriv_scale;
#         float dys = (next.at<sift_wt>(r+1, c) - next.at<sift_wt>(r-1, c) -
#                      prev.at<sift_wt>(r+1, c) + prev.at<sift_wt>(r-1, c))*cross_deriv_scale;
#
#         Matx33f H(dxx, dxy, dxs,
#                   dxy, dyy, dys,
#                   dxs, dys, dss);
#
#         Vec3f X = H.solve(dD, DECOMP_LU);
#
#         xi = -X[2];
#         xr = -X[1];
#         xc = -X[0];
#
#         if( std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f )
#             break;
#
#         if( std::abs(xi) > (float)(INT_MAX/3) ||
#             std::abs(xr) > (float)(INT_MAX/3) ||
#             std::abs(xc) > (float)(INT_MAX/3) )
#             return false;
#
#         c += cvRound(xc);
#         r += cvRound(xr);
#         layer += cvRound(xi);
#
#         if( layer < 1 || layer > nOctaveLayers ||
#             c < SIFT_IMG_BORDER || c >= img.cols - SIFT_IMG_BORDER  ||
#             r < SIFT_IMG_BORDER || r >= img.rows - SIFT_IMG_BORDER )
#             return false;
#     }
#
#     // ensure convergence of interpolation
#     if( i >= SIFT_MAX_INTERP_STEPS )
#         return false;
#
#     {
#         int idx = octv*(nOctaveLayers+2) + layer;
#         const Mat& img = dog_pyr[idx];
#         const Mat& prev = dog_pyr[idx-1];
#         const Mat& next = dog_pyr[idx+1];
#         Matx31f dD((img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1))*deriv_scale,
#                    (img.at<sift_wt>(r+1, c) - img.at<sift_wt>(r-1, c))*deriv_scale,
#                    (next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c))*deriv_scale);
#         float t = dD.dot(Matx31f(xc, xr, xi));
#
#         contr = img.at<sift_wt>(r, c)*img_scale + t * 0.5f;
#         if( std::abs( contr ) * nOctaveLayers < contrastThreshold )
#             return false;
#
#         // principal curvatures are computed using the trace and det of Hessian
#         float v2 = img.at<sift_wt>(r, c)*2.f;
#         float dxx = (img.at<sift_wt>(r, c+1) + img.at<sift_wt>(r, c-1) - v2)*second_deriv_scale;
#         float dyy = (img.at<sift_wt>(r+1, c) + img.at<sift_wt>(r-1, c) - v2)*second_deriv_scale;
#         float dxy = (img.at<sift_wt>(r+1, c+1) - img.at<sift_wt>(r+1, c-1) -
#                      img.at<sift_wt>(r-1, c+1) + img.at<sift_wt>(r-1, c-1)) * cross_deriv_scale;
#         float tr = dxx + dyy;
#         float det = dxx * dyy - dxy * dxy;
#
#         if( det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det )
#             return false;
#     }
#
#     kpt.pt.x = (c + xc) * (1 << octv);
#     kpt.pt.y = (r + xr) * (1 << octv);
#     kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5)*255) << 16);
#     kpt.size = sigma*powf(2.f, (layer + xi) / noctavelayers)*(1 << octv)*2;
#     kpt.response = std::abs(contr);
#
#     return true;
# }
