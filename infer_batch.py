import torch
# from src.helper_functions.helper_functions import parse_args
from src.models import create_model
import argparse
import numpy as np
from PIL import Image
from src.models.tresnet import TResNet


def infer_batch(model, classes_list, inputs, threshold=0.7):
    # inputs: batch, channel, height, weight
    print('ASL Example Inference code on a batch of images')

    output = torch.sigmoid(model(inputs))

    probs = output.cpu().detach().numpy()
    labels = []
    labels_probs = []

    # print(type(classes_list))
    # numpy array

    for i in range(0, inputs.shape[0]):
        np_output = probs[i, :]
        print(np_output.shape)
        detected_classes = classes_list[np_output > threshold]
        print(detected_classes)
        labels.append(detected_classes)
        labels_probs.append(np_output[np_output > threshold])

    return probs, labels, labels_probs


def load_model(model_type):

    if model_type is "L":
        model_name = "tresnet_l"
        path = './pth_files/MS_COCO_TRresNet_L_448_86.6.pth'
        input_size = 448
        threshold = 0.5
    elif model_type is "XL":
        model_name = "tresnet_xl"
        path = './pth_files/MS_COCO_TResNet_xl_640_88.4.pth'
        input_size = 640
        threshold = 0.5

    state = torch.load(path, map_location='cpu')
    num_classes = state['num_classes']

    if model_type is "L":
        do_bottleneck_head = False
        model = TResNet(layers=[4, 5, 18, 3], num_classes=num_classes, in_chans=3, width_factor=1.2,
                        do_bottleneck_head=do_bottleneck_head)
    elif model_type is "XL":
        model = TResNet(layers=[4, 5, 24, 3], num_classes=num_classes, in_chans=3, width_factor=1.3)

    model = model.cuda()
    model.load_state_dict(state['model'], strict=True)
    model.eval()

    classes_list = np.array(list(state['idx_to_class'].values()))

    return model, input_size, threshold, num_classes, classes_list


def test_single():

    model, input_size, threshold, num_classes, classes_list = load_model("L")

    pic_path = './pics/000000000885.jpg'

    im = Image.open(pic_path)
    im_resize = im.resize((input_size, input_size))
    np_img = np.array(im_resize, dtype=np.uint8)
    np_imgs = np.stack([np_img, np_img])
    print(np_imgs.shape)
    tensor_img = torch.from_numpy(np_imgs).permute(0, 3, 1, 2).float() / 255.0  # HWC to CHW

    tensor_batch = tensor_img.cuda()

    probs, labels, labels_probs = infer_batch(model, classes_list, tensor_batch, threshold)
    print(probs)
    print(labels)
    print(labels_probs)


def test_with_loader():

    model, input_size, threshold, num_classes, classes_list = load_model("L")


    from torchvision.transforms import transforms
    from data_loader import CocoObject
    from torch.autograd import Variable
    from sklearn.metrics import average_precision_score
    import torch.nn as nn
    from tqdm import tqdm as tqdm
    # crop_size = 224
    # image_size = 256
    batch_size = 12
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    val_transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        # transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        ])

    # Data samplers.
    ann_dir = '/home/ytianas/EMSE_COCO/cocodataset/annotations'
    image_dir = '/home/ytianas/EMSE_COCO/cocodataset/'
    test_data = CocoObject(ann_dir=ann_dir, image_dir=image_dir,
                           split='test', transform=val_transform)
    image_ids = test_data.image_ids
    image_path_map = test_data.image_path_map
    # 80 objects
    id2object = test_data.id2object
    id2labels = test_data.id2labels
    # Data loaders / batch assemblers.
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              shuffle=False, num_workers=4,
                                              pin_memory=True)
    count = 0
    yhats = []
    labels = []
    imagefiles = []
    res = list()

    t = tqdm(test_loader, desc = 'testing')

    for batch_idx, (images, objects, image_ids) in enumerate(t):

        images = Variable(images).cuda()
        objects = Variable(objects).cuda()

        # print(images.shape)

        object_preds = model(images)
        m = nn.Sigmoid()
        object_preds_r = m(object_preds)

        count = count + len(image_ids)
        for i in range(len(image_ids)):
            image_file_name = image_path_map[int(image_ids[i])]
            yhat = []
            label = id2labels[int(image_ids[i])]

            for j in range(len(object_preds[i])):
                a = object_preds_r[i][j].cpu().data.numpy()
                if a > threshold:
                    yhat.append(id2object[j])

            yhats.append(yhat)
            labels.append(label)
            imagefiles.append(image_file_name)


        res.append((image_ids, object_preds.data.cpu(), objects.data.cpu()))
        if count % 1000 == 0:
            print("count: " + str(count))

    preds_object = torch.cat([entry[1] for entry in res], 0)
    targets_object = torch.cat([entry[2] for entry in res], 0)
    eval_score_object = average_precision_score(targets_object.numpy(), preds_object.numpy())
    print('\nmean average precision of object classifier on test data is {}\n'.format(eval_score_object))


if __name__ == '__main__':

    # test_single()
    test_with_loader()