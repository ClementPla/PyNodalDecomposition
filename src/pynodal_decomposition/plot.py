from typing import Optional

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def imshow(
    img: np.ndarray,
    segment: Optional[np.ndarray] = None,
    border_width: int = 3,
    border_alpha: float = 1.0,
    title: Optional[str] = None,
    cmap: str = "jet_r",
    vmin: Optional[int] = None,
    vmax: Optional[int] = None,
    normalize: bool = False,
    ax: Optional[Axes] = None,
    border_cmap=None,
    show: bool = False,
    mask: Optional[np.ndarray] = None,
    heatmap: Optional[np.ndarray] = None,
    heatmap_alpha=0.5,
    binarize_heatmap: bool = False,
    show_segment_id: bool = False,
    threshold=0.5,
    font_size=8,
):
    """
    Display an image with optional segmentation overlay of the superpixels' borders.
    """
    segment = segment if segment is not None else None
    img = img.astype(np.float32)

    if normalize:
        img = (img - img.min()) / (img.max() - img.min())

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    if mask is not None:
        mask = 1 - mask.astype(np.float32)
        if img.ndim == 3:
            mask = np.expand_dims(mask, axis=2)
            img = np.concatenate([img, mask], axis=2)
            mask = None

    alphas = mask
    ax.imshow(img, cmap=cmap, interpolation="none", vmin=vmin, vmax=vmax, alpha=alphas)

    if heatmap is not None:
        if heatmap.ndim == 4:
            heatmap = heatmap.squeeze()
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap = heatmap > threshold if binarize_heatmap else heatmap
        ax.imshow(
            heatmap,
            cmap=cmap,
            alpha=(heatmap > threshold) * heatmap_alpha,
            interpolation="nearest",
        )

    if segment is not None:
        if show_segment_id:
            x_segment, y_segment = get_segments_centroids(segment)
            for n, (x, y) in enumerate(zip(x_segment, y_segment)):
                # n = segment[0, y, x].item()
                ax.text(
                    x,
                    y,
                    str(n),
                    fontsize=font_size,
                    color="white",
                    ha="center",
                    va="center",
                )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_width, border_width))

        segment_border = (cv2.morphologyEx(segment.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0).astype(np.float32)
        alphas = segment_border * border_alpha
        ax.imshow(segment_border, alpha=alphas, cmap=border_cmap)

    ax.set_axis_off()
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.margins(x=0)
    if title is not None:
        ax.set_title(title)
    if show:
        fig.tight_layout()
        fig.show()

    return fig


def get_segments_centroids(segment: np.ndarray):
    xs = []
    ys = []
    h, w = segment.shape
    labels = np.unique(segment)
    for label in labels:
        y, x = np.where(segment == label)
        xs.append(np.median(x))
        ys.append(np.median(y))

    return xs, ys
