### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ aff807a0-5124-11ed-1675-b74542d32634
begin
	using TestImages
	using ImageFiltering
	using ImageCore
	using PlutoUI
	using Interpolations
	using ImageShow
	using FileIO
	using ImageTransformations
	using OffsetArrays
	using Memoize
	using Parameters
	using Setfield
end

# ╔═╡ adcd3dca-878d-422f-9fb2-3004f7fec702
using PaddedViews

# ╔═╡ 483bc2c2-248d-48aa-8ba3-1c9f5c747e1f
using Base.Threads

# ╔═╡ c034f315-93fb-4b27-b274-6b2cb132e667
using Profile

# ╔═╡ 7b812cb5-95b1-403e-b613-5bfd116ae797
using ProfileCanvas

# ╔═╡ a270e66b-f2c7-4651-a5b3-3bd617583966
@memoize function fib(n)
	if n < 2
		return n
	else
		fib(n - 1) + fib(n - 2)
	end
end

# ╔═╡ 32290568-bad5-49bf-b3a2-83b65db24b17
@time fib(1000)

# ╔═╡ c8c71b3e-343e-4e9f-b44b-ce7b203faa39
TableOfContents()

# ╔═╡ 92685aab-2c03-451d-a7ed-ef2896023ea4
md"# Input"

# ╔═╡ 153288b8-3622-4b9c-a4d6-9b8d25accc20
@bind image_name PlutoUI.Select((@. first(splitext(TestImages.remotefiles))), default="lighthouse")

# ╔═╡ 0320abdb-f8d2-456f-a59a-7238dc11bf39
@bind image_file FilePicker()

# ╔═╡ bb892b57-5302-446f-bca9-a3c74c32c772
image, orig_image = let
	image = testimage(image_name)
	if !isnothing(image_file)
		io = IOBuffer()
		write(io, image_file["data"])
		image = load(Stream{format"PNG"}(io))
	end
	orig_image = image
	image = Gray.(image)
	image, orig_image
end

# ╔═╡ 0a603cf2-f83c-47b7-9d75-abafada7beee
md"# Overall"

# ╔═╡ 50e1198d-a432-4746-853c-b9970c77c1bf
2^(1/NUm)

# ╔═╡ 026d1ac7-ecbb-4cc8-bb3e-403a9ff628d3
to_gray(image) = @. gray(Gray(image))

# ╔═╡ 12399059-2f00-45f9-aff4-e75f48ea8ab3
to_gray(image)

# ╔═╡ 5de0b8f6-18ff-4cfb-93e4-ad94d765e630
imfilter(to_gray(image),  convert.(N0f8, Kernel.gaussian(1.2)))

# ╔═╡ 1d783473-dddc-43aa-8126-a8c8ad6ef9a3
md"# Scale space"

# ╔═╡ 23419689-7cfe-4a97-b917-928bc13dc766
Maybe{T} = Union{Nothing, T}

# ╔═╡ 9568b235-fa4b-4297-a3c8-0085317a9eea
@with_kw struct SIFT
	σ::Float32 = 1.6
	num_samples::Int = 3
	num_scales::Maybe{Int} = nothing
end

# ╔═╡ 26fe385f-543b-485e-935c-8556e23c86d7
@with_kw struct SIFTResult
	sift::SIFT
	num_scales::Maybe{Int} = nothing
	scales::Maybe{Vector} = nothing
	gauss_kernels = nothing
	gauss_images = nothing
	dog_images = nothing
	
end

# ╔═╡ 89c22646-6a92-4be5-8060-f059b54187f4
function get_num_octaves(image)
	convert(Int, log(image |> size |> minimum) / log(2) - 1 |> floor)
end

# ╔═╡ 1ba24b1d-e421-4d6b-9f8e-234ffa6221d6
num_samples = 3

# ╔═╡ 32ad1d8f-299d-4a2c-80ea-cdee9b48a694
# function gen_scale_space_images(image::T, σ=1.6, num_samples=3) where T
# 	num_octaves = get_num_octaves(image)
# 	kernel = Kernel.gaussian(σ)
# 	h, w = size(image)
# 	# Compute first octave
# 	first_octave = T[]
# 	sizehint!(first_octave, num_samples)
# 	for i in 1:num_samples
# 		image = imfilter(image, kernel)
# 		push!(first_octave, image)
# 	end
	
# 	# Compute other octaves
# 	rest_octaves = [
# 		[
# 		begin
# 			step = 2 * n
# 			image = image[begin:step:end, begin:step:end]
# 			imresize(image, (h, w), method=Linear())
# 		end
# 		for image in first_octave
# 		]
# 		for n in 2:num_octaves
# 	]

# 	dogs = vcat(
# 		diff(first_octave),
# 		[diff(nth_octave) for nth_octave in rest_octaves]...)
# end

# ╔═╡ 08aad0cf-5780-4736-b92d-54b13d936ecf
Kernel.gaussian(2^8)

# ╔═╡ 9a4a24e8-5e21-4743-b0ca-1b269c65b013
[(a, b) for a in 'a':'z' for b in 1:3]

# ╔═╡ 10b5099a-a084-45e1-8d87-700df026dcda
Vector{Int}(undef, 3)

# ╔═╡ 8e6bff68-c239-48d4-89d9-50f67d15d72d
function gen_single_octave(image, kernel, num_samples)
	images = Vector{typeof(image)}(undef, num_samples)
	for i in 1:num_samples
		image = imfilter(image, kernel)
		images[i] = image
	end
	images
end

# ╔═╡ 33c84d54-4026-4ce9-957d-50a0034ed6c1
function run_sift(image::Matrix, sift = SIFT())::SIFTResult
	grayed = to_gray(image)
	result = SIFTResult(sift=sift)
	@set! result.num_scales = num_scales = get_num_octaves(grayed)
	k = 2^(1/num_scales)
	@set! result.scales = [sift.σ * (k^(n-1)) for n in 1:result.num_scales]
	FType = (eltype(grayed))
	@set! result.gauss_kernels = [
		begin
			k = Kernel.gaussian(k)
		end for k in result.scales]
	result
	@set! result.gauss_images = [
		gen_single_octave(grayed, kernel, sift.num_samples) for
		kernel in result.gauss_kernels
	]
end

# ╔═╡ 4e1cf4ca-cbc7-4570-a350-4f47076bcc8c
result = let 
	@code_warntype run_sift(image)
	result = run_sift(image)
end

# ╔═╡ 5c0def39-bba8-461c-b5a7-a1c591de0f5f
function gen_scale_space_images(image::T, σ=1.6, num_samples=3) where T
	num_octaves = get_num_octaves(image)
	h, w = size(image)
	k = 2^(1/num_samples)
	octaves = [
		begin
			kernel = Kernel.gaussian(σ * k^n)
			gen_single_octave(image, kernel, num_samples) 
		end
	for n in 0:num_octaves-1]
	dogs = diff.(octaves)
	vcat(dogs...)
	# Compute first octave
end

# ╔═╡ 6cb7fec3-d921-4085-9a09-0828dcb4e2e4
dogs = gen_scale_space_images(image, 1.6, num_samples)

# ╔═╡ 95a3d3ff-fdbd-4b4f-aeee-def2910fbf33
size(dogs[end]), length(dogs)

# ╔═╡ 45a94fb5-86c4-48bf-8cbc-4eca3768e47d
num_samples

# ╔═╡ da1b98f9-753d-4219-9270-c6ae9c1f7d47
md"# Local extrema detection"

# ╔═╡ dd5a2e9d-9cf8-437e-966c-90c6e37bad59
function is_maxima(curr, prev, next, x, y, scale, threshold=0.03)
	value = curr[x, y] - threshold
	result = (value > prev[x, y]) & (value > next[x, y])
	result = result && all(
		value > prev[x+i,y+j] && value > next[x+i,y+j] && value > curr[x+i,y+j]
		for (i, j) in Iterators.product((-scale, scale), (-scale, scale))
	)
end

# ╔═╡ 2bde6807-2b5a-4622-8817-b0729abaacf6
function is_minima(curr, prev, next, x, y, scale, threshold=0.03)
	value = curr[x, y] + threshold * scale
	result = (value < prev[x, y]) & (value < next[x, y])
	result = result && all(
		value < prev[x+i,y+j] && value < next[x+i,y+j] && value < curr[x+i,y+j]
		for (i, j) in Iterators.product((-scale, scale), (-scale, scale))
	)
end

# ╔═╡ 71a73d82-b837-467b-95ff-3a5b2f6a32ac
div(1, 3)

# ╔═╡ e0da518e-ede1-4eea-8373-7533836d89a4
function detect_local_extrema(dogs, num_samples::Int, threshold=0.03)
	h, w = size(dogs |> first)
	extremas = fill(false, (h, w))
	loop = 1
	for (prev, curr, next) in zip(dogs, dogs[begin+1:end], dogs[begin+2:end])
		scale = 1
		for (x, y) in Iterators.product(1+scale:h-scale, 1+scale:w-scale)
			maxima = is_maxima(curr, prev, next, x, y, scale, threshold)
			minima = is_minima(curr, prev, next, x, y, scale, threshold) 
			extremas[x, y] = extremas[x, y] || maxima || minima
		end
		loop += 1
	end
	extremas
end

# ╔═╡ 765f5e36-8047-4c73-90f9-c63c572e6245
bgray(x) = broadcast(gray ∘ Gray, x)

# ╔═╡ b2ddfa2d-4e10-447e-9fd1-3441eb27ed55


# ╔═╡ 23631994-af57-4ad9-a45a-4d91b225ab43
bdogs = bgray.(dogs);

# ╔═╡ 77a8c80e-cb08-4805-921d-3650f5b96e9f
@time detect_local_extrema(bdogs, num_samples)

# ╔═╡ b6aad84e-2b2b-4514-aaa1-3a14247ed78d
mask = detect_local_extrema(bgray.(dogs), num_samples)

# ╔═╡ 88dabe26-6d1d-4c15-9667-5523e2e3c7e0
let 
	image = copy(orig_image)
	image[mask] .= RGB(1, 0, 0)
	image
end

# ╔═╡ 3245b2f7-0f9d-443a-a280-474c84d6bae8
findall(mask)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
ImageCore = "a09fc81d-aa75-5fe9-8630-4744c3626534"
ImageFiltering = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
ImageShow = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
ImageTransformations = "02fcd773-0e25-5acc-982a-7f6622650795"
Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
Memoize = "c03570c3-d221-55d1-a50c-7939bbd78826"
OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
PaddedViews = "5432bcbf-9aad-5242-b902-cca2824c8663"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Profile = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
ProfileCanvas = "efd6af41-a80b-495e-886c-e51b0c7d77a3"
Setfield = "efcf1570-3423-57d1-acb7-fd33fddbac46"
TestImages = "5e47fb64-e119-507b-a336-dd2b206d9990"

[compat]
FileIO = "~1.16.0"
ImageCore = "~0.8.22"
ImageFiltering = "~0.6.21"
ImageShow = "~0.3.1"
ImageTransformations = "~0.8.13"
Interpolations = "~0.13.6"
Memoize = "~0.4.4"
OffsetArrays = "~1.12.8"
PaddedViews = "~0.5.11"
Parameters = "~0.12.3"
PlutoUI = "~0.7.44"
ProfileCanvas = "~0.1.6"
Setfield = "~1.1.1"
TestImages = "~1.7.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.2"
manifest_format = "2.0"
project_hash = "7d3a6389943dac7e5c6e8428aa925b4f63ccdfc9"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "1dd4d9f5beebac0c03446918741b1a03dc5e5788"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.6"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e7ff6cadf743c098e08fca25c91103ee4303c9bb"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "3ca828fe1b75fa84b021a7860bd039eaea84d2f2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.3.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "fb21ddd70a051d882a1686a5a550990bbe371a95"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.1"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "681ea870b918e7cff7111da58791d7f718067a19"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.2"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "c36550cb29cbe373e95b3f40486b9a4148f89ffd"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.2"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "7be5f99f7d15578798f338f5433b6c432ea8037b"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "78e2c69783c9753a91cdae88a8d432be85a2ab5e"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.0+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IdentityRanges]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be8fcd695c4da16a1d6d0cd213cb88090a150e3b"
uuid = "bbac6d45-d8f3-5730-bfe4-7a449cd117ca"
version = "0.3.1"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "db645f20b59f060d8cfae696bc9538d13fd86416"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.8.22"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ColorVectorSpace", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageCore", "LinearAlgebra", "OffsetArrays", "Requires", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "bf96839133212d3eff4a1c3a80c57abc7cfbf0ce"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.6.21"

[[deps.ImageIO]]
deps = ["FileIO", "Netpbm", "PNGFiles"]
git-tree-sha1 = "0d6d09c28d67611c68e25af0c2df7269c82b73c7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.4.1"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils", "Libdl", "Pkg", "Random"]
git-tree-sha1 = "5bc1cb62e0c5f1005868358db0692c994c3a13c6"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.2.1"

[[deps.ImageMagick_jll]]
deps = ["Artifacts", "Ghostscript_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "124626988534986113cfd876e3093e4a03890f58"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.12+3"

[[deps.ImageShow]]
deps = ["Base64", "FileIO", "ImageCore", "OffsetArrays", "Requires", "StackViews"]
git-tree-sha1 = "832abfd709fa436a562db47fd8e81377f72b01f9"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.1"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "ColorVectorSpace", "CoordinateTransformations", "IdentityRanges", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "e4cc551e4295a5c96545bb3083058c24b78d4cf0"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.8.13"

[[deps.IndirectArrays]]
git-tree-sha1 = "c2a145a145dc03a7620af1444e0264ef907bd44f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "0.5.1"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "b7bc05649af456efc75d178846f47006c2c4c3c7"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.6"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "3f91cd3f56ea48d4d2a75c2a65455c5fc74fa347"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.3"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "94d9c52ca447e23eac0c0f074effbcd38830deb5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.18"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2ce8695e1e699b68702c03402672a69f54b8aca9"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.2.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Memoize]]
deps = ["MacroTools"]
git-tree-sha1 = "2b1dfcba103de714d31c033b5dacc2e4a12c7caa"
uuid = "c03570c3-d221-55d1-a50c-7939bbd78826"
version = "0.4.4"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.Netpbm]]
deps = ["ColorVectorSpace", "FileIO", "ImageCore"]
git-tree-sha1 = "09589171688f0039f13ebe0fdcc7288f50228b52"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "f71d8950b724e9ff6110fc948dff5a329f901d64"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.8"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "f809158b27eba0c18c269cf2a2be6ed751d3e81d"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.17"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "03a7a85b76381a3d04c7a1656039197e70eda03d"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.11"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "6c01a9b494f6d2a9fc180a08b182fcb06f0958a0"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.4.2"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "efc140104e6d0ae3e7e30d56c98c4a927154d684"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.48"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProfileCanvas]]
deps = ["Base64", "JSON", "Pkg", "Profile", "REPL"]
git-tree-sha1 = "e42571ce9a614c2fbebcaa8aab23bbf8865c624e"
uuid = "efd6af41-a80b-495e-886c-e51b0c7d77a3"
version = "0.1.6"

[[deps.Quaternions]]
deps = ["DualNumbers", "LinearAlgebra", "Random"]
git-tree-sha1 = "4ab19353944c46d65a10a75289d426ef57b0a40c"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.5.7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays", "Statistics"]
git-tree-sha1 = "3d52be96f2ff8a4591a9e2440036d4339ac9a2f7"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.3.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "5d65101b2ed17a8862c4c05639c3ddc7f3d791e1"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.8.7"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "f86b3a049e5d05227b10e15dbb315c5b90f14988"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.9"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StringDistances]]
deps = ["Distances", "StatsAPI"]
git-tree-sha1 = "ceeef74797d961aee825aabf71446d6aba898acb"
uuid = "88034a9c-02f8-509d-84a9-84ec65e18404"
version = "0.11.2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TestImages]]
deps = ["AxisArrays", "ColorTypes", "FileIO", "ImageIO", "ImageMagick", "OffsetArrays", "Pkg", "StringDistances"]
git-tree-sha1 = "03492434a1bdde3026288939fc31b5660407b624"
uuid = "5e47fb64-e119-507b-a336-dd2b206d9990"
version = "1.7.1"

[[deps.TiledIteration]]
deps = ["OffsetArrays"]
git-tree-sha1 = "5683455224ba92ef59db72d10690690f4a8dc297"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.3.1"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═aff807a0-5124-11ed-1675-b74542d32634
# ╠═a270e66b-f2c7-4651-a5b3-3bd617583966
# ╠═32290568-bad5-49bf-b3a2-83b65db24b17
# ╠═c8c71b3e-343e-4e9f-b44b-ce7b203faa39
# ╟─92685aab-2c03-451d-a7ed-ef2896023ea4
# ╟─153288b8-3622-4b9c-a4d6-9b8d25accc20
# ╠═0320abdb-f8d2-456f-a59a-7238dc11bf39
# ╠═bb892b57-5302-446f-bca9-a3c74c32c772
# ╠═0a603cf2-f83c-47b7-9d75-abafada7beee
# ╠═9568b235-fa4b-4297-a3c8-0085317a9eea
# ╠═26fe385f-543b-485e-935c-8556e23c86d7
# ╠═50e1198d-a432-4746-853c-b9970c77c1bf
# ╠═33c84d54-4026-4ce9-957d-50a0034ed6c1
# ╠═4e1cf4ca-cbc7-4570-a350-4f47076bcc8c
# ╠═026d1ac7-ecbb-4cc8-bb3e-403a9ff628d3
# ╠═12399059-2f00-45f9-aff4-e75f48ea8ab3
# ╠═5de0b8f6-18ff-4cfb-93e4-ad94d765e630
# ╟─1d783473-dddc-43aa-8126-a8c8ad6ef9a3
# ╠═23419689-7cfe-4a97-b917-928bc13dc766
# ╠═89c22646-6a92-4be5-8060-f059b54187f4
# ╠═1ba24b1d-e421-4d6b-9f8e-234ffa6221d6
# ╠═32ad1d8f-299d-4a2c-80ea-cdee9b48a694
# ╠═08aad0cf-5780-4736-b92d-54b13d936ecf
# ╠═9a4a24e8-5e21-4743-b0ca-1b269c65b013
# ╠═10b5099a-a084-45e1-8d87-700df026dcda
# ╠═8e6bff68-c239-48d4-89d9-50f67d15d72d
# ╠═5c0def39-bba8-461c-b5a7-a1c591de0f5f
# ╠═6cb7fec3-d921-4085-9a09-0828dcb4e2e4
# ╠═95a3d3ff-fdbd-4b4f-aeee-def2910fbf33
# ╠═45a94fb5-86c4-48bf-8cbc-4eca3768e47d
# ╟─da1b98f9-753d-4219-9270-c6ae9c1f7d47
# ╠═adcd3dca-878d-422f-9fb2-3004f7fec702
# ╠═dd5a2e9d-9cf8-437e-966c-90c6e37bad59
# ╠═2bde6807-2b5a-4622-8817-b0729abaacf6
# ╠═483bc2c2-248d-48aa-8ba3-1c9f5c747e1f
# ╠═71a73d82-b837-467b-95ff-3a5b2f6a32ac
# ╠═e0da518e-ede1-4eea-8373-7533836d89a4
# ╠═765f5e36-8047-4c73-90f9-c63c572e6245
# ╠═b2ddfa2d-4e10-447e-9fd1-3441eb27ed55
# ╟─c034f315-93fb-4b27-b274-6b2cb132e667
# ╠═7b812cb5-95b1-403e-b613-5bfd116ae797
# ╠═23631994-af57-4ad9-a45a-4d91b225ab43
# ╠═77a8c80e-cb08-4805-921d-3650f5b96e9f
# ╠═b6aad84e-2b2b-4514-aaa1-3a14247ed78d
# ╠═88dabe26-6d1d-4c15-9667-5523e2e3c7e0
# ╠═3245b2f7-0f9d-443a-a280-474c84d6bae8
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
