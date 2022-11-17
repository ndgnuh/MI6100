function compute_scale_space(image::Matrix)
end

function is_valid_keypoint(height::Int, width::Int, num_layers::Int,
                           row::Int, col::Int, layer::Int)
    return (row < 2 ||
            col < 2 ||
            layer < 2 ||
            row > height - 1 ||
            col > width - 1 ||
            layer > num_layers - 1)
end
