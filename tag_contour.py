import cv2


def find_largest_contour_idx(contours, hierarchy, first_child_idx):
    """
    first_child_idx specifies the first contour in the level we want to find largest contour of
    """
    # return None if no contour is in the list or the given first child is not in the list
    if first_child_idx >= hierarchy.shape[1]:
        return None

    i = first_child_idx
    largest_idx = -1
    largest_area = -1
    while i != -1:
        area = cv2.contourArea(contours[i])
        if area > largest_area:
            largest_idx = i
            largest_area = area

        # update i
        cntinfo = hierarchy[0][i]  # (next, prev, first_child, parent)
        i = cntinfo[0]

    return largest_idx


