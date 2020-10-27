from pycocotools.coco import COCO
# from PIL import Image
from data_loader import CocoObject
import numpy as np
import os
import cv2
from tqdm import tqdm

if __name__ == '__main__':
    ann_dir = '/home/ytianas/EMSE_COCO/cocodataset/annotations'
    image_dir = '/home/ytianas/EMSE_COCO/cocodataset/'
    test_data = CocoObject(ann_dir=ann_dir, image_dir=image_dir,
                           split='test', transform=None)

    image_ids = test_data.image_ids
    image_path_map = test_data.image_path_map
    # 80 objects
    id2object = test_data.id2object
    id2labels = test_data.id2labels

    # print(id2labels)
    # print(id2object)
    # exit(-1)

    ann_cat_name = test_data.ann_cat_name
    ann_cat_id = test_data.ann_cat_id
    bboxes = test_data.bbox
    masks = test_data.mask

    fill_values = [0, 127, 255]

    count = 0

    t = tqdm(image_ids)

    for image_id in t:
        anns = ann_cat_id[image_id]
        bbox = bboxes[image_id]
        mask = masks[image_id]
        path = image_path_map[image_id]

        unique_anns = list(set(anns))
        # print(unique_anns)
        for unique_ann in unique_anns:

            output_filename = "{}_{}.jpg".format(path[0:-4], unique_ann)
            # COCO_val2014_000000240972.jpg
            # mask_to_union =[]
            union_mask = np.zeros_like(mask[0])

            for i in range(0, len(anns)):
                if anns[i] == unique_ann:
                    # mask_to_union.append(mask[i])
                    union_mask = np.add(union_mask, mask[i])

            union_mask = union_mask > 0

            image_path = os.path.join(image_dir, "val2014", path)
            image = cv2.imread(image_path)
            for fill_value in fill_values:
                obj_image = image.copy()
                obj_image[np.nonzero(union_mask)] = fill_value
                bg_image = image.copy()
                bg_image[np.nonzero(1 - union_mask)] = fill_value

                cv2.imwrite("../coco_img/obj2_{}/{}".format(fill_value, output_filename), obj_image)
                cv2.imwrite("../coco_img/bg2_{}/{}".format(fill_value, output_filename), bg_image)

        count += 1
        # if count >= 10:
        #     break