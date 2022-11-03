
function offset(idx, dim, by)
    for (d, b) in zip(dim, by)
        idx = @set idx[d] = idx[d] + b
    end
    return idx
end

function im_gradients(img)
    dims = ndims(img)
    grads = map(1:dims) do d
        shape = Tuple(i == d ? 3 : 1 for i in 1:dims)
        kernel = reshape([-1 // 2, 0, 1 // 2], shape...)
        imfilter(x, kernel)
    end
end

function im_gradients_hessians(img)
    dims = ndims(img)
    grads = im_gradients(img)
    hessians = [copy(img) for _ in 1:dims, _ in 1:dims]
    for i in 1:dims
        hesses = im_gradients(grads[i])
        for j in 1:dims
            hessians[i, j] .= hesses[j]
        end
    end
    grads, hessians
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

function compute_hessians_at(arr, idx::CartesianIndex)
    n = length(idx)
    map(product(1:n, 1:n)) do (d1, d2)
        i1 = offset(idx, (d1, d2), (1, 1))
        i2 = offset(idx, (d1, d2), (1, -1))
        i3 = offset(idx, (d1, d2), (-1, 1))
        i4 = offset(idx, (d1, d2), (-1, -1))
        (arr[i1] - arr[i2] - arr[i3] + arr[i4]) / 4
    end
end
