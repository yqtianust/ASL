import random
import numpy as np
random.seed(0)
import pickle

sample_sizes = [(275, 356), (319, 364), (352, 348)]
models = ['XL']

for i in range(0, 1):
    model_name = models[i]
    result = np.load("./mlc_all_probs_analyze_distribution/{}.npz".format(model_name))
    ground_truth_acc = result['acc']
    violate_obj = result["violate_obj"]
    violate_bg = result["violate_bg"]
    # violate_both = result["violate_both"]

    size_1, size_2 = sample_sizes[i]

    sampled_obj = []
    sampled_bg = []

    while len(sampled_obj) < size_1:
        random_number = random.randrange(0, len(violate_obj))
        if violate_obj[random_number]:
                if random_number not in sampled_obj:
                    sampled_obj.append(random_number)

    while len(sampled_bg) < size_2:
        random_number = random.randrange(0, len(violate_bg))
        if violate_bg[random_number]:
            if random_number not in sampled_bg:
                sampled_bg.append(random_number)

    gt = pickle.load(open("COCO_ground_truth.pickle", "rb"))
    names = sorted(gt.keys())

    # create_folder("download_resample/obj2_255")
    # create_folder("download_resample/obj2_0")
    with open("cp_img_resampe.sh", 'w') as f:
        for k in sampled_obj:
            img = names[k]
            f.write("cp ../coco_img/org/{} download_sample/obj2_255/{}_org.JPEG \n".format(img, img))
            f.write("cp ../coco_img/merged_obj2_255/{} download_sample/obj2_255/ \n".format(img))

    # create_folder("download_sample/bg2_255")
    with open("cp_img_resampe.sh", 'a+') as f:
        for k in sampled_bg:
            img = names[k]
            f.write("cp ../coco_img/org/{} download_sample/bg2_255/{}_org.JPEG \n".format(img, img))
            f.write("cp ../coco_img/merged_bg2_255/{} download_sample/bg2_255/ \n".format(img))

    a = pickle.load(open("id2object.pickle", 'rb'))
    obj_classes = {}
    for k in range(0, len(sampled_obj)):
        key = names[sampled_obj[k]]
        y_gt = gt[key]
        sample_class = []
        for j, value in enumerate(y_gt):
            if value == 1:
                sample_class.append(a[j])
        obj_classes[sampled_obj[k]] = "_".join(sample_class)

    bg_classes = {}
    for k in range(0, len(sampled_bg)):
        key = names[sampled_bg[k]]
        y_gt = gt[key]
        sample_class = []
        for j, value in enumerate(y_gt):
            if value == 1:
                sample_class.append(a[j])
        bg_classes[sampled_bg[k]] = "_".join(sample_class)

    with open("download_sample/download_sample_mlc.csv".format(model_name), 'a+') as f:
        for k in range(0, len(sampled_obj)):
            filename = "{},{},{},{}\n".format(names[sampled_obj[k]], model_name, obj_classes[sampled_obj[k]], "obj")
            f.write(filename)
        for k in range(0, len(sampled_bg)):
            filename = "{},{},{},{}\n".format(names[sampled_bg[k]], model_name, bg_classes[sampled_bg[k]], "bg")
            f.write(filename)


