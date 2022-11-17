using Chain
using ImageCore

function show_pyramid(pyr)
    layer, height, width = size(pyr[1])
    @chain begin
        map(pyr) do octave
            map(eachslice(octave; dims=1)) do layer
                @chain begin
                    convert(Matrix{Gray}, layer)
                    transpose(_)
                end
            end
        end
        reduce(hcat, _)
        transpose(_)
    end
end
