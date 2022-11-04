### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ d11befb8-89b4-46ba-b6f4-2f7e7807308f
using Pkg

# ╔═╡ aef080e2-5892-11ed-0046-effd5c5eb84a
# # This cell disables Pluto's package manager and activates the global environment. Click on ? inside the bubble next to Pkg.activate to learn more.
# # (added automatically because a sysimage is used)
Pkg.activate()

# ╔═╡ 0c41a03d-fb65-4d20-8b1c-841e615d4dab
using Images

# ╔═╡ bf2148d9-340a-4a18-8860-2ec6bf16d3fe
using PyCall

# ╔═╡ 1e860c2a-859f-42eb-93b3-63afd0dc80d7
using ImageShow

# ╔═╡ aef080fe-5892-11ed-267e-f300b486eaea
import OpenCV as cv2

# ╔═╡ 65cc9824-0edd-4cbe-a4a1-1d1fd02ac086
plt = pyimport("matplotlib.pyplot")

# ╔═╡ fbcadc43-2aef-4c46-a2f0-d98e2d4ca0da
# cv2 = pyimport("cv2")


# ╔═╡ a1ee19bf-3288-44a4-a373-868d959e7c9a
cv2

# ╔═╡ d579e88c-dbb2-47b1-9490-a7075e4682b2
ImageShow

# ╔═╡ 2bfe4d75-6066-484d-ae2c-caea27dab94f
x = let img = cv2.imread("./../samples/box.png")
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = permutedims(img[1, :, :], (2, 1))
	Gray.(img/ 255)
end

# ╔═╡ 37396b21-98bc-418c-8dc6-7482e27d0178
sift = cv2.xfeatures2d.SIFT_create

# ╔═╡ 91d080e5-e995-4ec5-b376-97d616c54671
keypoint = cv2.KeyPoint()

# ╔═╡ 007d0c65-b6f6-4a70-b2dc-243b5a3733d5
keypoint.pt

# ╔═╡ Cell order:
# ╠═d11befb8-89b4-46ba-b6f4-2f7e7807308f
# ╠═aef080e2-5892-11ed-0046-effd5c5eb84a
# ╠═0c41a03d-fb65-4d20-8b1c-841e615d4dab
# ╠═bf2148d9-340a-4a18-8860-2ec6bf16d3fe
# ╠═aef080fe-5892-11ed-267e-f300b486eaea
# ╠═1e860c2a-859f-42eb-93b3-63afd0dc80d7
# ╠═65cc9824-0edd-4cbe-a4a1-1d1fd02ac086
# ╠═fbcadc43-2aef-4c46-a2f0-d98e2d4ca0da
# ╠═a1ee19bf-3288-44a4-a373-868d959e7c9a
# ╠═d579e88c-dbb2-47b1-9490-a7075e4682b2
# ╠═2bfe4d75-6066-484d-ae2c-caea27dab94f
# ╠═37396b21-98bc-418c-8dc6-7482e27d0178
# ╠═91d080e5-e995-4ec5-b376-97d616c54671
# ╠═007d0c65-b6f6-4a70-b2dc-243b5a3733d5
