using .Constants
using LinearAlgebra
function match(desc1, desc2)
    matches = Tuple{Int, Int}[]
    sizehint!(matches, length(desc1))
    for (idx1, d1) in enumerate(desc1)
        min_dist = Inf
        rest_min = Inf
        min_feature = 0
        for (idx2, d2) in enumerate(desc2)
            dist = norm(d1 - d2)
            if dist < min_dist
                min_dist = dist
                min_feature = idx2
            elseif
                rest_min = dist
            end
        end
        push!(matches, (idx1, min_feature))
    end
    if min_dist < rest_min * Constants.REL_DIST_MATCH_THRESH
        push!(matches, (idx1, idx2))
    end
    return matches
end
