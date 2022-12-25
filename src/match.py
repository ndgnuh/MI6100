import numpy as np
import cv2
from matplotlib.patches import ConnectionPatch
from matplotlib import pyplot as plt
from PIL import Image
from typing import List
from icecream import ic


def match_sift_features(
        features1,
        features2,
        descriptors1,
        descriptors2,
        rel_dist_match_thresh=0.6
):
    """ A brute force method for finding matches between two sets of SIFT features.

    Args:
        features1: A set of SIFT features.
        features2: A set of SIFT features.
        descriptor1: Corresponding descriptor set
        descriptor2: Corresponding descriptor set
    Returns:
        matches: A list of matches. Each match is a (feature, feature) tuples.
    """

    matches = list()
    mask = list()

    for idx1, (feature1, descriptor1) in enumerate(zip(features1, descriptors1)):
        min_dist = np.inf
        rest_min = np.inf
        min_feature = None

        for idx2, (feature2, descriptor2) in enumerate(zip(features2, descriptors2)):
            if idx2 in mask:
                continue

            dist = np.linalg.norm(descriptor1 - descriptor2)

            if dist < min_dist:
                min_dist = dist
                min_feature = feature2

            elif dist < rest_min:
                rest_min = dist

        if min_dist < rest_min * rel_dist_match_thresh:
            matches.append((feature1, min_feature))
            mask.append(idx2)

    return matches


def absolute_coordinate(kpt):
    y, x = kpt.pt
    return 0, int(x), int(y)


def if_object_found(matches, K1, K2, threshold=10):
    # num_match_kpts = len(set(k for k, _ in matches))
    num_match_kpts = len(matches)
    num_kpts = len(K1)
    ic(num_match_kpts, num_kpts, len(K2))
    return num_match_kpts >= threshold, num_match_kpts, num_kpts


def visualize_matches(
    matches: List,
    img1: np.ndarray,
    img2: np.ndarray,
    K1: List,
    K2: List,
    show: bool = False
):
    """ Plots SIFT keypoint matches between two images.

    Args:
        matches: A list of matches. Each match is a (feature, feature) tuples.
        img1: The image in which the first match features were found.
        img2: The image in which the second match features were found.
    """

    coords_1 = [absolute_coordinate(match[0]) for match in matches]
    coords_1y = [coord[1] for coord in coords_1]
    coords_1x = [coord[2] for coord in coords_1]
    coords_1xy = [(x, y) for x, y in zip(coords_1x, coords_1y)]

    coords_2 = [absolute_coordinate(match[1]) for match in matches]
    coords_2y = [coord[1] for coord in coords_2]
    coords_2x = [coord[2] for coord in coords_2]
    coords_2xy = [(x, y) for x, y in zip(coords_2x, coords_2y)]

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(img1, cmap='Greys_r')
    ax2.imshow(img2, cmap='Greys_r')

    ax1.scatter(coords_1x, coords_1y)
    ax2.scatter(coords_2x, coords_2y)

    found, matched, total = if_object_found(matches, K1, K2)
    plt.title(
        f"Found: {str(found):5s}, matched: {matched}/{total} keypoint(s)")

    for p1, p2 in zip(coords_1xy, coords_2xy):
        con = ConnectionPatch(xyA=p2, xyB=p1, coordsA="data",
                              coordsB="data", axesA=ax2, axesB=ax1, color="red")
        ax2.add_artist(con)

    if show:
        plt.show()
    return fig


def run_sift(image, max_image_size=None):
    # ensure PIL
    if isinstance(image, str):
        image = Image.open(image)

    if max_image_size is not None:
        image.thumbnail(max_image_size)

    sift = cv2.SIFT_create()
    image = image.convert("L")
    image = np.array(image)

    # Keypoints set and descriptors set
    K, D = sift.detectAndCompute(image, None)
    return image, K, D
