import bm3d
import cv2
import numpy as np
from skimage.filters import rank, threshold_otsu, threshold_yen
from skimage.morphology import disk, skeletonize
from skimage.segmentation import felzenszwalb, random_walker, relabel_sequential, slic


def preprocess(image: np.ndarray, sigma=100 / 255, contrast=1.0, gamma=1.0) -> np.ndarray:
    image = (
        bm3d.bm3d(
            image / 255,
            sigma_psd=sigma,
            stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING,
        ).copy(order="C")
        * contrast
    )
    image = image**gamma
    image = image - image.min()
    image = image / image.max()

    image = (255 * image).astype(np.uint8)
    # image = cv2.medianBlur(image, 7)
    return image


def get_segment(image, method, **parameters):
    match method:
        case "watershed":
            pass
        case "slic":
            segments = slic(image, **parameters)
        case "felzenszwalb":
            segments = felzenszwalb(image, **parameters)

    return segments


def close_small_gaps(segments, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closing = cv2.morphologyEx(segments.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    return closing


def get_mask(img, method="otsu", quantile=50, radius=9):
    if img.ndim == 3:
        img = img[:, :, 0]
    match method:
        case "otsu":
            threshold = threshold_otsu(img)
        case "yen":
            threshold = threshold_yen(img)
        case "quantile":
            threshold = np.percentile(img, quantile)
        case "local_otsu":
            footprint = disk(radius)
            return (rank.otsu(img, footprint=footprint) < 127).astype(np.uint8)

    # high_quartile = np.percentile(img, threshold)
    mask = img < threshold
    return mask


def mask_segments(segments, mask):
    segments[mask] = 0
    return segments.astype(np.uint8) * 255


def random_walker_threshold(img, segments, threshold=50):
    markers = np.zeros(segments.shape, dtype=np.uint)

    low_quartile = np.percentile(img, 5)
    high_quartile = np.percentile(img, 75)
    markers[img[:, :, 0] > high_quartile] = 2
    markers[img[:, :, 0] < low_quartile] = 1
    labels = random_walker(img[:, :, 0], markers, beta=2, mode="bf")

    segments[labels == 0] = 0
    return segments.astype(np.uint8) * 255


def remove_disconnected_elements(segments):
    output = np.zeros_like(segments)
    unique_value = np.unique(segments)
    for value in unique_value:
        if value == 0:
            continue
        mask = segments == value
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        argsort_area = np.argsort(stats[:, cv2.CC_STAT_AREA])

        mask = labels == argsort_area[-2]
        output[mask] = value
    return output


def merge_segments_from_circular_mask(segments, mask, circular_threshold=0.2, min_area=10):
    mask = mask.astype(np.uint8)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = False
    params.filterByConvexity = False
    params.filterByInertia = False

    params.filterByCircularity = True
    params.minCircularity = 1 - circular_threshold
    params.maxCircularity = 1 + circular_threshold
    params.minInertiaRatio = 0.7
    params.maxInertiaRatio = 1.00
    params.minArea = min_area
    blob_detector = cv2.SimpleBlobDetector_create(params)
    keypoints = blob_detector.detect((255 * (1 - mask)))
    if keypoints:
        connected_components = cv2.connectedComponents(mask)[1]

        for keypoint in keypoints:
            x, y = keypoint.pt
            x, y = int(x), int(y)
            val = connected_components[y, x]
            val_seg = segments[y, x]
            current_comp = connected_components == val
            contours, hierarchy = cv2.findContours(
                (1 - current_comp).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            if len(contours) > 2:
                continue
            segments[current_comp] = val_seg
    return segments


def fuse_inner_segments(segment):
    labels = np.unique(segment)
    areas = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for l in labels:
        if l == 0:
            continue
        mask = segment == l

        bounds = cv2.boundingRect(mask.astype(np.uint8))
        centroids = cv2.moments(mask.astype(np.uint8))

        x, y, w, h = bounds
        bounds_area = w * h

        flood_point = (
            int(centroids["m10"] / centroids["m00"]),
            int(centroids["m01"] / centroids["m00"]),
        )

        skeleton = skeletonize(mask)
        mask = cv2.floodFill(skeleton.astype(np.uint8), None, flood_point, 255)[1] > 0

        if mask.sum() > bounds_area:
            continue

        filled_area = mask.astype(np.uint8)

        filled_area = cv2.morphologyEx(filled_area, cv2.MORPH_OPEN, kernel)

        areas.append((l, bounds_area, filled_area))

    areas = sorted(areas, key=lambda x: x[1], reverse=False)
    new_segment = segment.copy()
    for label, area, filled_area in areas:
        new_segment[filled_area > 0] = label

    return relabel_sequential(new_segment)[0]
