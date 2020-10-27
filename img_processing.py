import numpy as np


# bbox: x, y, w, h
# mask: h x w x 3/1
# rect: h x w

def merge_ann(masks_to_merge):
    """

    :param masks_to_merge:  array of masks
    :return:
    """
    merge_rect = np.copy(masks_to_merge[0])
    for id in range(1, len(masks_to_merge)):
        merge_rect = np.add(merge_rect, masks_to_merge[id])

    merge_rect[merge_rect >= 1] = 1
    sum_over_x = np.sum(merge_rect, axis=1)
    y = np.where(sum_over_x >= 1)[0][0]
    sum_over_y = np.sum(merge_rect, axis=0)
    x = np.where(sum_over_y >= 1)[0][0]

    h = np.where(sum_over_x >= 1)[0][-1] - y
    w = np.where(sum_over_y >= 1)[0][-1] - x

    merge_bbox = (x, y, w, h)
    return merge_rect, merge_bbox


def iou_bbox(bbox_A, bbox_B):
    if _boxesIntersect(bbox_A, bbox_B) is False:
        return 0
    interArea = _getIntersectionArea(bbox_A, bbox_B)
    union = _getUnionAreas(bbox_A, bbox_B, interArea=interArea)
    # intersection over union
    iou = interArea / union
    assert iou >= 0
    return iou


def _getIntersectionArea(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    # intersection area
    return (xB - xA + 1) * (yB - yA + 1)


def _boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[0] + boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[0] + boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] + boxA[1] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3] + boxB[1]:
        return False  # boxA is below boxB
    return True


def _getUnionAreas(boxA, boxB, interArea=None):
    area_A = _getArea(boxA)
    area_B = _getArea(boxB)
    if interArea is None:
        interArea = _getIntersectionArea(boxA, boxB)
    return float(area_A + area_B - interArea)


def _getArea(box):
    return box[2] * box[3]


def iou_mask(mask_A, mask_B):
    intersect_area = np.bitwise_and(mask_A, mask_B).sum()
    iou = 0

    if intersect_area != 0:
        union_area = mask_A.sum() + mask_B.sum() - intersect_area
        iou = intersect_area / union_area

    return iou
