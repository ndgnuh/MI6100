from PIL import Image, ImageEnhance
from itertools import product

# https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html
src = "assets/queries/cup-in-scene.jpg"

Image.open(src)
contrasts = [0.5, 1, 1.5]
brightnesses = [0.2, 0.5, 1, 1.5]
for (c, b) in product(contrasts, brightnesses):
    if b == 1 and c == 1:
        continue
    img = Image.open(src)
    img = ImageEnhance.Contrast(img).enhance(c).convert("RGB")
    img = ImageEnhance.Brightness(img).enhance(b).convert("RGB")
    output = f"assets/queries/cup-in-scene-c{c:1.1f}-b{b:1.1f}.jpg"
    img.save(output)
    img.close()
    print(f"saved {output}")
