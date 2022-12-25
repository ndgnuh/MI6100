from src.match import match_sift_features, run_sift, visualize_matches
from icecream import ic
from matplotlib import pyplot as plt
from os import path, makedirs
from tqdm import tqdm
from argparse import ArgumentParser
from shutil import rmtree


def basename(p):
    return path.splitext(path.basename(p))[0]


def output_name(ref_path, query_path):
    br = basename(ref_path)
    bq = basename(query_path)
    return f"{br}@{bq}.jpg"


inputs = [
    ("assets/refs/box.png", "assets/queries/box_in_scene.png"),
    ("assets/refs/cup.jpg", "assets/queries/cup-in-scene.jpg"),
    ("assets/refs/cup.jpg", "assets/queries/cup-in-scene-c0.5-b0.5.jpg"),
    ("assets/refs/cup.jpg", "assets/queries/cup-in-scene-c0.5-b1.0.jpg"),
    ("assets/refs/cup.jpg", "assets/queries/cup-in-scene-c0.5-b1.5.jpg"),
    ("assets/refs/cup.jpg", "assets/queries/cup-in-scene-c1.0-b0.2.jpg"),
    ("assets/refs/cup.jpg", "assets/queries/cup-in-scene-c1.0-b0.5.jpg"),
    ("assets/refs/cup.jpg", "assets/queries/cup-in-scene-c1.0-b1.5.jpg"),
    ("assets/refs/cup.jpg", "assets/queries/cup-in-scene-c1.5-b0.5.jpg"),
    ("assets/refs/cup.jpg", "assets/queries/cup-in-scene-c1.5-b1.0.jpg"),
    ("assets/refs/cup.jpg", "assets/queries/cup-in-scene-c1.5-b1.5.jpg"),
    ("assets/refs/cup.jpg", "assets/queries/cup-view-point-1.jpg"),
    ("assets/refs/cup.jpg", "assets/queries/cup-view-point-2.jpg"),
    ("assets/refs/tokyo_tower.jpg", "assets/queries/tokyo-tower.jpg"),
    ("assets/refs/hust.png", "assets/queries/hust-01.jpg"),
    ("assets/refs/hust.png", "assets/queries/hust-02.jpg"),
    ("assets/refs/eiffel.jpg", "assets/queries/eiffel.png"),
]

parser = ArgumentParser()
parser.add_argument("-f", dest="force", action="store_true", default=False)
parser.add_argument("-t", dest="thumbnail", default="1024x1024")
args = parser.parse_args()
thumbnail = tuple(map(int, args.thumbnail.split("x")))
output_dir = f"assets/results-{args.thumbnail}"

# Process output dir
if args.force:
    try:
        rmtree(output_dir)
    except Exception:
        pass
makedirs(output_dir, exist_ok=True)
print(thumbnail)

for (ref, query) in inputs:
    I1, K1, D1 = run_sift(ref, (512, 512))
    I2, K2, D2 = run_sift(query, thumbnail)
    output_file = output_name(ref, query)
    output_file = path.join(output_dir, output_file)

    if path.isfile(output_file) and not args.force:
        print(f"Found output {output_file}, skipping")
        continue
    matches = match_sift_features(K1, K2, D1, D2)
    fig = visualize_matches(matches, I1, I2, K1, K2)
    fig.savefig(output_file)
    # ic(matches)
