from pycocotools.coco import COCO
# from PIL import Image
from data_loader_RAM import CocoObject
import numpy as np
import os
import cv2
from tqdm import tqdm

if __name__ == '__main__':
    ann_dir = '/home/ytianas/EMSE_COCO/cocodataset/annotations'
    image_dir = '/home/ytianas/EMSE_COCO/cocodataset/'
    test_data = CocoObject(ann_dir=ann_dir, image_dir=image_dir,
                           split='val', transform=None)
    # image_ids = test_data.image_ids
    image_path_map = test_data.image_path_map
    # 80 objects
    id2object = test_data.id2object
    id2labels = test_data.id2labels

    # print(id2labels)
    # print(id2object)
    # exit(-1)

    # ann_cat_name = test_data.ann_cat_name
    # ann_cat_id = test_data.ann_cat_id
    # bboxes = test_data.bbox
    # masks = test_data.mask


    print("start counting")
    count = 0

    t = tqdm(range(0, len(test_data.image_ids)))

    result = {}

    for idx in t:
        # anns = ann_cat_id[image_id]
        # bbox = bboxes[image_id]
        img, _, image_id, anns, ann_cat_name, bboxes, mask = test_data.__getitem__(idx)
        # mask = masks[image_id]
        path = image_path_map[image_id]

        # unique_anns = list(set(anns))
        # print(unique_anns)
        # for unique_ann in unique_anns:

        output_filename = "{}.jpg".format(path[0:-4])
        # COCO_val2014_000000240972.jpg
        # mask_to_union =[]
        # print(len(anns))
        # print(len(mask))

        if len(anns) > 0:

            union_mask = np.zeros_like(mask[0])

            for i in range(0, len(anns)):
                union_mask = np.add(union_mask, mask[i])

            union_mask = union_mask > 0

            image_path = os.path.join(image_dir, "val2014", path)
            image = cv2.imread(image_path)

            img_size = image.shape[0] * image.shape[1]
            obj_size = np.sum(np.sum(union_mask))

            ratio = obj_size * 1.0 / img_size
            count += 1
        else:
            print("No mask: {}".format(output_filename))

            ratio = 0.0
        # if count >= 10:
        #     break

        result[path] = ratio

    keys_sorted = sorted(result.keys())
    np_arr = np.zeros([len(keys_sorted)])
    for i in range(0, len(keys_sorted)):

        key = keys_sorted[i]
        np_arr[i] = result[key]
        if i == 0:
            print(key)
            print(np_arr[i])

    np.save("mlc_obj_size_ratio.npy", np_arr)