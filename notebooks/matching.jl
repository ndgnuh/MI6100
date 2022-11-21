### A Pluto.jl notebook ###
# v0.19.15

using Markdown
using InteractiveUtils

# ╔═╡ daa9c826-6986-11ed-3306-0f036220687d
using Pkg

# ╔═╡ 631dba0c-cb23-4e45-ab53-9f5e4dc60ef9
Pkg.activate("..")

# ╔═╡ f257483d-c95d-416a-87ad-40e5eb26de4a
using SIFT

# ╔═╡ 2c1b9ed1-4742-45c2-99e4-39fa9e679bbb
using ImageCore

# ╔═╡ 8383fa56-f837-40dd-9f62-8de9f2b09ba1
using FileIO

# ╔═╡ 563bdfad-c7a7-43bd-a471-9a21824430dc
image1 = load("../samples/box.png")

# ╔═╡ 115a3329-e838-4e63-9916-8eaeaad6da65
image2 = load("../samples/box_in_scene.png")

# ╔═╡ c3faae78-8f5d-42e1-baaa-c9b1c5ccd300
kpt1 = SIFT.sift(image1)

# ╔═╡ 3a614423-b22a-4eb4-ab4c-11a238b44dd0
kpt2 = SIFT.sift(image2)

# ╔═╡ b3fd092e-7c14-444e-b4c2-ef5f317b6338
matches = SIFT.match(kpt1, kpt2)

# ╔═╡ Cell order:
# ╠═daa9c826-6986-11ed-3306-0f036220687d
# ╠═631dba0c-cb23-4e45-ab53-9f5e4dc60ef9
# ╠═f257483d-c95d-416a-87ad-40e5eb26de4a
# ╠═2c1b9ed1-4742-45c2-99e4-39fa9e679bbb
# ╠═8383fa56-f837-40dd-9f62-8de9f2b09ba1
# ╠═563bdfad-c7a7-43bd-a471-9a21824430dc
# ╠═115a3329-e838-4e63-9916-8eaeaad6da65
# ╠═c3faae78-8f5d-42e1-baaa-c9b1c5ccd300
# ╠═3a614423-b22a-4eb4-ab4c-11a238b44dd0
# ╠═b3fd092e-7c14-444e-b4c2-ef5f317b6338
