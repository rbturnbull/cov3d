import cv2
import numpy as np


def _bbox_overlap_ok(contour1, contour2):
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)

    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))

    if x_overlap > 0.2 * min(w1, w2):
        return False

    return True


def segment_slice(vol, z):
    im = vol[z].astype(np.uint8)

    # Apply thresholding to create a binary image
    _, binary = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply inverse binary thresholding to focus on dark regions
    _, binary_cropped_inv = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours on the masked inverted binary image
    contours, _ = cv2.findContours(binary_cropped_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # exclude any contours which run the entire width of the slice
    def _filter_func(contour):
        if cv2.contourArea(contour) < 500:
            return False
        i, j, w, h = cv2.boundingRect(contour)
        if w >= 0.8 * im.shape[0]:
            return False
        return True

    sorted_contours = list(filter(_filter_func, sorted_contours))[:2]

    # remove overlaps
    filtered_contours = []
    for ii, contour_i in enumerate(sorted_contours):
        exclude = False
        for jj, contour_j in enumerate(sorted_contours):
            if ii != jj and not _bbox_overlap_ok(contour_i, contour_j):
                exclude = True
        if not exclude:
            filtered_contours.append(contour_i)

    # resort again by left edge
    sorted_contours = sorted(
        filtered_contours,
        key=lambda c: cv2.boundingRect(c)[0],
    )

    edges = [-1, -1]

    # Throw away any invalid results
    if len(sorted_contours) < 2:
        return edges

    # Calc bounding boxes around the two largest contours and work out the edges

    for ii, contour in enumerate(sorted_contours):
        i, j, w, h = cv2.boundingRect(contour)
        if ii == 0:
            edges[0] = i + w
        if ii == 1:
            edges[1] = i

    return edges


def segment_volumes(vol):
    edges = np.vstack([segment_slice(vol, z) for z in range(vol.shape[0])])
    sel = np.s_[int(0.3 * edges.shape[0]) : int(0.7 * edges.shape[0])]
    left = edges[sel, 0]
    left = left[left < int(edges.shape[1] * 0.8)].max()
    right = edges[sel, 1]
    right = right[right > int(edges.shape[1] * 0.2)].min()

    return vol[:, :, :left] / 255.0, vol[:, :, right:] / 255.0
