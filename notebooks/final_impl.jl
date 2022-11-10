### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ 8f4afee0-5d18-11ed-2a03-83aa239f0f84
using LRUCache

# ╔═╡ 4c563b39-065f-492b-a3dd-ebb45f26b295
using PlutoLinks

# ╔═╡ 59307eae-fd67-49d3-98a3-588d4b172340
using ImageIO

# ╔═╡ a9370d52-0ca3-4fb6-b69b-5e8c2c65e213
using FileIO

# ╔═╡ 7eea1deb-fc6e-4f1a-89d1-42e6ff451804
using ImageCore

# ╔═╡ 802e9ef8-b3b3-42e3-b313-363a5d236865
using ImageShow

# ╔═╡ 3c874c50-1128-44d3-95b5-b3ed460efd58
using OffsetArrays

# ╔═╡ ebc25a46-b08e-4793-b66b-e9d40b8da443
Draw = let
	using ImageDraw
	@ingredients("../SIFT/draw.jl").Draw
end

# ╔═╡ 5c16d793-4e0a-43dc-987d-2bb8ffefcd5b
using Chain

# ╔═╡ 29200873-e263-45b2-8ada-0dacd79d34f5
S = let
	using ImageFiltering

	using StaticArrays
	using Parameters
	using Setfield
	using UnPack
	using Accessors
	using ImageTransformations
	using Memoize
	@ingredients("../SIFT/struct.jl")
	@ingredients("../SIFT/gaussian.jl")
	@ingredients("../SIFT/sift.jl")
end

# ╔═╡ af3787c9-cc02-4999-9b94-763967383bcc
using BenchmarkTools

# ╔═╡ cbfdbb4c-f3e0-450b-b0ea-1eec1526a6c3
using Random

# ╔═╡ d4d261c8-1522-4ce6-bea1-2b848cc4e42f
using DataStructures

# ╔═╡ 260ab7c7-8743-4040-8d4b-6860f1240dbc
using LinearAlgebra

# ╔═╡ b588a012-2a8c-43da-83ed-e5f3f3965408
# ╠═╡ disabled = true
#=╠═╡
using StaticArrays
  ╠═╡ =#

# ╔═╡ b51a3b28-1b94-45ec-9038-3ece5a48a33d
@ingredients("../SIFT/sift.jl")

# ╔═╡ 5d125872-ec3e-4788-9020-9df08cc4bbc7
@SVector zeros(Float32, 3)

# ╔═╡ 906037fa-3603-4b4b-8e57-fecf33bfc210
@MVector zeros(3)

# ╔═╡ 67bff29b-0796-401e-8d40-73624b5fb51a
n = 2

# ╔═╡ 8132a218-28b9-4eaa-abc4-5849a811a2ff
m= 3

# ╔═╡ a4c577ab-2963-4024-8081-e45b41d39a49
SMatrix{n,m}(i for i in 1:n, j in 1:m)

# ╔═╡ c3a6814f-33ee-4c7b-9e66-8390cd07032e
function dynamic2(n)
	x = Deque{Int}()
	# sizehint!(x, n)
	for i in 1:n
		if rand([true, false])
			x = push!(x, i)
		end
	end
	x
end


# ╔═╡ f2983f37-ab88-4a4b-8329-6a3a9a6cd265
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
let
	Random.seed!(0)
	@btime dynamic2(20000)
end
  ╠═╡ =#

# ╔═╡ 4361e5e0-7f12-4fa5-8aed-ebf8c306d67d
mutable struct LinkedList{T}
	value::T
	index::Int
	next::Union{Nothing, LinkedList}
	prev::Union{Nothing, LinkedList}
	function LinkedList(value::T) where T
		lst = new{T}(value, 1, nothing, nothing)
		lst.next = lst
		lst.prev = lst
		lst
	end
end

# ╔═╡ 340ee4e6-714a-47e2-8f63-06d2791d1ff1
methodswith(list() |> typeof)

# ╔═╡ a0a5cc94-2bfc-47f9-9f64-40bcda4b56b3
function f1(n)
	l2 = MutableLinkedList{Int}()
end

# ╔═╡ 9537618d-7379-44f4-9299-ebaa668de4e9


# ╔═╡ 52d7a918-a6eb-44e4-896b-489d5e19981d


# ╔═╡ 079156fc-513c-4cd5-ba15-d53abe5d7751


# ╔═╡ 54805733-420d-4489-81bf-8d405c8fbaa3


# ╔═╡ f30ca828-4822-4d1a-9f7d-b3632d3b1c56


# ╔═╡ cba302d6-e01f-420d-aa36-a5e6cd16c295


# ╔═╡ 966737b2-c718-486d-b10e-27e7230c6b2b


# ╔═╡ b9c3c7b1-ca62-49cf-bdb9-0c90931aa112


# ╔═╡ c5a9ccb5-8eaa-450c-b143-83cd684ab790


# ╔═╡ 4d1eb2d4-b1a4-4dc8-80af-3d67a3249c53
function Base.length(l::LinkedList)
	orig_idx = l.index
	len = 1
	cur = l.next
	while orig_idx != cur.index
		cur = l.next
		len += 1
	end
	len
end

# ╔═╡ a834c6ea-06f6-4514-81cd-cc7b0fa2146b
function dynamic(n)
	x = Int[]
	# sizehint!(x, n)
	for i in 1:n
		if rand([true, false])
			push!(x, i)
		end
	end
	resize!(x, length(x))
end

# ╔═╡ 4ccfa22f-6776-456f-94cd-98184e44c04f
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
let
	Random.seed!(0)
	@btime dynamic(20000)
end
  ╠═╡ =#

# ╔═╡ cfb684bd-9c32-4126-8266-f0f540ef232c
length(l2)

# ╔═╡ 0cf896fe-c543-40a3-8ad6-6f4cc63763d2
l =  LinkedList(2)

# ╔═╡ 306e0a0c-7c21-4d94-bd2d-32d8fdaabcce
length(l)

# ╔═╡ c016a338-8a52-4d2c-9f06-9dc4df8ef098


# ╔═╡ 09b5e2eb-b4c8-44b4-9492-f07b2d1f8956
L

# ╔═╡ edb6e742-cdd3-4d9e-bd38-a979f19589f8
rad2

# ╔═╡ 18111ae3-71f3-48e5-a79c-43ebb11714c5


# ╔═╡ 323dc098-9199-4681-9f8d-9dda0d75973a
g = S.Hessian(rand(3, 3))

# ╔═╡ 78ca30a2-c9de-457b-99bf-2b60fdd88dcf
input_image = "../samples/box.png"

# ╔═╡ be1186af-891d-4a0c-a0c9-379234d1f0b4
kpts = let image = load(input_image)
	S.sift(image, 1.6, 3, 0.5)
end

# ╔═╡ 59b4878b-e334-478c-9dba-bcd4fc3816ff
let image = RGB.(load(input_image))
	Draw.draw_keypoints!(image, kpts, RGB(0, 1, 0))
end

# ╔═╡ 00d7843e-ef48-45c3-b6f4-27ff1c713d1d
# ╠═╡ disabled = true
#=╠═╡
sift = let sift = S.SIFT()
	image = reinterpret(N0f8, load(input_image))
	sift = S.fit(sift, image)
	sift.keypoints
	sift
end
  ╠═╡ =#

# ╔═╡ 4401cbf1-91fe-4bcf-8f79-7da4986f8bc3
#=╠═╡
sift.base_image
  ╠═╡ =#

# ╔═╡ b277219f-db5d-42a8-830a-8f9d46448a26
# ╠═╡ disabled = true
#=╠═╡
hessian = Dict(
	octave => S.Hessian(sift.dpyr[octave])
	for octave in 1:8
);
  ╠═╡ =#

# ╔═╡ 3b368e6b-1d11-491b-a1d5-15542ead67d8
# ╠═╡ disabled = true
#=╠═╡
gradient = Dict(
	octave => S.Gradient(sift.dpyr[octave])
	for octave in 1:8
);
  ╠═╡ =#

# ╔═╡ 334d5d38-d5bb-4bd6-a841-8ca92c9133b6
function localize_keypoint(s, gradient, hessian, octave, layer, row, col)
	dog = s.dpyr[octave]
	grad = gradient[octave]
	hess = hessian[octave]
	mlayer::Int, mrow::Int, mcol::Int = size(grad.x)

	# Candidate property
	low_contrast::Bool = true
	on_edge::Bool = true
	outside::Bool = false

	
	# Localize via quadratic fit
	x = @MVector Float32[layer, row, col]
	x̂ = @MVector zeros(Float32, 3)
	converge = false
	for i = 1:10
		layer, row, col = trunc.(Int, x)

		# Check if the coordinate is inside the image
		if (layer < 2 || layer > mlayer - 1
			|| row < 2 || row > mrow - 1
			|| col < 2 || col > mcol - 1)
			outside = true
			break
		end

		# Solve for x̂
		g = grad(layer, row, col) 
		h = hess(layer, row, col)
		if det(h) == 0
			break
		end
		x̂ .= -inv(h) * g

		# Check if the changes is small
		if any(@. abs(x̂) <= 0.5)
			converge = true
			break
		end

		# Update coordinate
		x .= x + x̂
	end

	# True coordinate (maybe)
	layer, row, col = trunc.(Int, x)

	# More check if localized
	contrast::Float32 = 0
	if converge
		# Calculate contrast
		g = grad(layer, row, col) 
		contrast =  dog[layer, row, col] + sum(g .* x̂) / 2
		low_contrast = abs(contrast) * mlayer < 0.04
		
		# Calcuate edge response
		h::Matrix{Float32} = @view hess(layer, row, col)[2:3, 2:3]
		dh::Float32 = det(h)
		th::Float32 = tr(h)
		r = 10
		on_edge = th^2 * r >= dh * (r + 1)^2
	end

	# Calculate angle if
	# keypoint is high contrast and
	# keypoint is not on any edge
	sigma = 1
	kptsize = s.sigma * (2^(layer - 1 / (mlayer - 3)) * (2^(octave - 1)))
	
	(
		outside=outside,
		size=3,
		converge=converge, 
		contrast=abs(contrast),
		low_contrast=low_contrast,
		on_edge=on_edge,
		layer=layer, 
		angle = nothing,
		row=trunc(Int, row * 2.0^(octave - 2)),
		col=trunc(Int, col * 2.0^(octave - 2))
	) # row = row, col=col
end

# ╔═╡ c4c99a3a-c6c6-4cde-b681-66d25a861625
#=╠═╡
let
	kpt = sift.keypoints[400]
	octave = kpt.octave
	layer = kpt.layer
	row = kpt.row
	col = kpt.col
	localize_keypoint(sift, gradient, hessian, octave, layer, row, col)
end
  ╠═╡ =#

# ╔═╡ 0bf8b317-e22f-4d25-9d6c-759bb662c67c
@chain begin
	map(sift.keypoints) do kpt
		octave = kpt.octave
		layer = kpt.layer
		row = kpt.row
		col = kpt.col
		localize_keypoint(sift, gradient, hessian, octave, layer, row, col)
	end
	filter(_) do kpt
		kpt.converge && !kpt.outside && !kpt.low_contrast && !kpt.on_edge
	end
end

# ╔═╡ 2926d7da-248f-491b-93d3-5d0e457ed323


# ╔═╡ 8954eab8-c7e1-47df-98d6-bd1d09b47cb3


# ╔═╡ 720e2ce6-2b5c-48dd-97f3-282672528990
let image = RGB.(load(input_image))
	Draw.draw_keypoints!(image, kpts, RGB(1, 0, 0))
end

# ╔═╡ 2e73cc5d-bccc-4fb3-94b1-f4cd51a63f06


# ╔═╡ 1f6b109f-9b8b-4e30-8dc4-4786a62e8007


# ╔═╡ 5b9c9ba3-3bc1-43c8-a1b8-038f775ba360
findlocalmaxima(rand(9, 9, 10), window=(3, 3, 3))

# ╔═╡ 20ce2d1b-2c26-4de6-8838-c6780c52342a
# ╠═╡ disabled = true
#=╠═╡
using Chain
  ╠═╡ =#

# ╔═╡ 7d5eecc1-a144-49f7-b378-498bbcedd166
#=╠═╡
kpt = sift.keypoints[1]
  ╠═╡ =#

# ╔═╡ f6fda39e-c72d-41e7-b4c4-f68a8cc51e41
let 
	@unpack row = kpt
	print(row)
end

# ╔═╡ dd6091da-7801-4fb7-878b-935101ebfa98
function show_pyramid(pyr)
	layer, height, width = size(pyr[1])
	@chain begin
		map(pyr) do octave
			map(eachslice(octave, dims=1)) do layer
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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Accessors = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Chain = "8be319e6-bccf-4806-a6f7-6fae938471bc"
DataStructures = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
ImageCore = "a09fc81d-aa75-5fe9-8630-4744c3626534"
ImageDraw = "4381153b-2b60-58ae-a1ba-fd683676385f"
ImageFiltering = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
ImageIO = "82e4d734-157c-48bb-816b-45c225c6df19"
ImageShow = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
ImageTransformations = "02fcd773-0e25-5acc-982a-7f6622650795"
LRUCache = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Memoize = "c03570c3-d221-55d1-a50c-7939bbd78826"
OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Setfield = "efcf1570-3423-57d1-acb7-fd33fddbac46"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
UnPack = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"

[compat]
Accessors = "~0.1.22"
BenchmarkTools = "~1.3.2"
Chain = "~0.5.0"
DataStructures = "~0.18.13"
FileIO = "~1.16.0"
ImageCore = "~0.9.4"
ImageDraw = "~0.2.5"
ImageFiltering = "~0.7.2"
ImageIO = "~0.6.6"
ImageShow = "~0.3.6"
ImageTransformations = "~0.9.5"
LRUCache = "~1.3.0"
Memoize = "~0.4.4"
OffsetArrays = "~1.12.8"
Parameters = "~0.12.3"
PlutoLinks = "~0.1.5"
Setfield = "~1.1.1"
StaticArrays = "~1.5.9"
UnPack = "~1.0.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.2"
manifest_format = "2.0"
project_hash = "437f7c867f849432ceb0a190d4bd77604fbf1fa1"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.Accessors]]
deps = ["Compat", "CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Requires", "Test"]
git-tree-sha1 = "eb7a1342ff77f4f9b6552605f27fd432745a53a3"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.22"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["ArrayInterfaceCore", "Compat", "IfElse", "LinearAlgebra", "Static"]
git-tree-sha1 = "d6173480145eb632d6571c148d94b9d3d773820e"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "6.0.23"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "732cddf5c7a3d4e7d4829012042221a724a30674"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.24"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "d9a9701b899b30332bbcb3e1679c41cce81fb0e8"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.2"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.Chain]]
git-tree-sha1 = "8c4920235f6c561e401dfe569beb8b924adad003"
uuid = "8be319e6-bccf-4806-a6f7-6fae938471bc"
version = "0.5.0"

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

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "cc4bd91eba9cdbbb4df4746124c22c0832a460d6"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.1.1"

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

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

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

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[deps.ImageDraw]]
deps = ["Distances", "ImageCore", "LinearAlgebra"]
git-tree-sha1 = "6ed6e945d909f87c3013e391dcd3b2a56e48b331"
uuid = "4381153b-2b60-58ae-a1ba-fd683676385f"
version = "0.2.5"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "8b251ec0582187eff1ee5c0220501ef30a59d2f7"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.2"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "342f789fd041a55166764c351da1710db97ce0e0"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.6"

[[deps.ImageShow]]
deps = ["Base64", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "b563cf9ae75a635592fc73d3eb78b86220e55bd8"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.6"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "ColorVectorSpace", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "8717482f4a2108c9358e5c3ca903d3a6113badc9"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.9.5"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "5cd07aab533df5170988219191dfad0519391428"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.3"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "842dd89a6cb75e02e85fdd75c760cdc43f5d6863"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.6"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

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

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "a77b273f1ddec645d1b7c4fd5fb98c8f90ad10a5"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.1"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "0f960b1404abb0b244c1ece579a0ec78d056a5d1"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.15"

[[deps.LRUCache]]
git-tree-sha1 = "d64a0aff6691612ab9fb0117b0995270871c5dfc"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.3.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

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

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "dedbebe234e06e1ddad435f5c6f4b85cd8ce55f7"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.2.2"

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
deps = ["FileIO", "ImageCore"]
git-tree-sha1 = "18efc06f6ec36a8b801b23f076e3c6ac7c3bf153"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.2"

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

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

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

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f6cf8e7944e50901594838951729a1861e668cb8"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.2"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "0e8bcc235ec8367a8e9648d48325ff00e4b0a545"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.5"

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

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "fcebf40de9a04c58da5073ec09c1c1e95944c79b"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.6.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

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

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "dad726963ecea2d8a81e26286f625aee09a91b7c"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.4.0"

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays", "Statistics"]
git-tree-sha1 = "793b6ef92f9e96167ddbbd2d9685009e200eb84f"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.3.3"

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

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "8fb59825be681d451c246a795117f317ecbcaa28"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.2"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "de4f0a4f049a4c87e4948c04acff37baf1be01a6"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.7.7"

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

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

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

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "70e6d2da9210371c927176cb7a56d41ef1260db7"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.1"

[[deps.TiledIteration]]
deps = ["ArrayInterface", "OffsetArrays"]
git-tree-sha1 = "5e02b75701f1905e55e44fc788bd13caedb5a6e3"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.4.1"

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

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

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
# ╠═8f4afee0-5d18-11ed-2a03-83aa239f0f84
# ╠═4c563b39-065f-492b-a3dd-ebb45f26b295
# ╠═59307eae-fd67-49d3-98a3-588d4b172340
# ╠═a9370d52-0ca3-4fb6-b69b-5e8c2c65e213
# ╠═7eea1deb-fc6e-4f1a-89d1-42e6ff451804
# ╠═802e9ef8-b3b3-42e3-b313-363a5d236865
# ╠═b588a012-2a8c-43da-83ed-e5f3f3965408
# ╠═3c874c50-1128-44d3-95b5-b3ed460efd58
# ╠═ebc25a46-b08e-4793-b66b-e9d40b8da443
# ╠═b51a3b28-1b94-45ec-9038-3ece5a48a33d
# ╠═5c16d793-4e0a-43dc-987d-2bb8ffefcd5b
# ╠═29200873-e263-45b2-8ada-0dacd79d34f5
# ╠═5d125872-ec3e-4788-9020-9df08cc4bbc7
# ╠═906037fa-3603-4b4b-8e57-fecf33bfc210
# ╠═67bff29b-0796-401e-8d40-73624b5fb51a
# ╠═8132a218-28b9-4eaa-abc4-5849a811a2ff
# ╠═a4c577ab-2963-4024-8081-e45b41d39a49
# ╠═be1186af-891d-4a0c-a0c9-379234d1f0b4
# ╠═a834c6ea-06f6-4514-81cd-cc7b0fa2146b
# ╠═c3a6814f-33ee-4c7b-9e66-8390cd07032e
# ╠═af3787c9-cc02-4999-9b94-763967383bcc
# ╠═cbfdbb4c-f3e0-450b-b0ea-1eec1526a6c3
# ╠═f2983f37-ab88-4a4b-8329-6a3a9a6cd265
# ╠═4ccfa22f-6776-456f-94cd-98184e44c04f
# ╠═4361e5e0-7f12-4fa5-8aed-ebf8c306d67d
# ╠═d4d261c8-1522-4ce6-bea1-2b848cc4e42f
# ╠═340ee4e6-714a-47e2-8f63-06d2791d1ff1
# ╠═a0a5cc94-2bfc-47f9-9f64-40bcda4b56b3
# ╠═cfb684bd-9c32-4126-8266-f0f540ef232c
# ╠═9537618d-7379-44f4-9299-ebaa668de4e9
# ╠═52d7a918-a6eb-44e4-896b-489d5e19981d
# ╠═079156fc-513c-4cd5-ba15-d53abe5d7751
# ╠═54805733-420d-4489-81bf-8d405c8fbaa3
# ╠═f30ca828-4822-4d1a-9f7d-b3632d3b1c56
# ╠═cba302d6-e01f-420d-aa36-a5e6cd16c295
# ╠═966737b2-c718-486d-b10e-27e7230c6b2b
# ╠═b9c3c7b1-ca62-49cf-bdb9-0c90931aa112
# ╠═c5a9ccb5-8eaa-450c-b143-83cd684ab790
# ╠═4d1eb2d4-b1a4-4dc8-80af-3d67a3249c53
# ╠═0cf896fe-c543-40a3-8ad6-6f4cc63763d2
# ╠═306e0a0c-7c21-4d94-bd2d-32d8fdaabcce
# ╠═c016a338-8a52-4d2c-9f06-9dc4df8ef098
# ╠═09b5e2eb-b4c8-44b4-9492-f07b2d1f8956
# ╠═edb6e742-cdd3-4d9e-bd38-a979f19589f8
# ╠═59b4878b-e334-478c-9dba-bcd4fc3816ff
# ╠═18111ae3-71f3-48e5-a79c-43ebb11714c5
# ╠═323dc098-9199-4681-9f8d-9dda0d75973a
# ╠═78ca30a2-c9de-457b-99bf-2b60fdd88dcf
# ╠═00d7843e-ef48-45c3-b6f4-27ff1c713d1d
# ╠═4401cbf1-91fe-4bcf-8f79-7da4986f8bc3
# ╠═b277219f-db5d-42a8-830a-8f9d46448a26
# ╠═3b368e6b-1d11-491b-a1d5-15542ead67d8
# ╠═260ab7c7-8743-4040-8d4b-6860f1240dbc
# ╠═334d5d38-d5bb-4bd6-a841-8ca92c9133b6
# ╠═c4c99a3a-c6c6-4cde-b681-66d25a861625
# ╠═0bf8b317-e22f-4d25-9d6c-759bb662c67c
# ╠═2926d7da-248f-491b-93d3-5d0e457ed323
# ╠═8954eab8-c7e1-47df-98d6-bd1d09b47cb3
# ╠═720e2ce6-2b5c-48dd-97f3-282672528990
# ╠═2e73cc5d-bccc-4fb3-94b1-f4cd51a63f06
# ╠═1f6b109f-9b8b-4e30-8dc4-4786a62e8007
# ╠═5b9c9ba3-3bc1-43c8-a1b8-038f775ba360
# ╠═20ce2d1b-2c26-4de6-8838-c6780c52342a
# ╠═7d5eecc1-a144-49f7-b378-498bbcedd166
# ╠═f6fda39e-c72d-41e7-b4c4-f68a8cc51e41
# ╠═dd6091da-7801-4fb7-878b-935101ebfa98
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
