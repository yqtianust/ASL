import torch
from src.helper_functions.helper_functions import parse_args
from src.models import create_model
import argparse
import numpy as np
from PIL import Image


def infer_batch(model, classes_list, inputs, threshold=0.7):
    # inputs: batch, channel, height, weight
    print('ASL Example Inference code on a batch of images')

    output = torch.sigmoid(model(tensor_batch))

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


if __name__ == '__main__':
    # arg parser:
    parser = argparse.ArgumentParser(description='ASL MS-COCO Inference on a single image')

    parser.add_argument('--model_path', type=str, default='./pth_files/TRresNet_L_448_86.6.pth')
    parser.add_argument('--pic_path', type=str, default='./pics/000000000885.jpg')
    parser.add_argument('--model_name', type=str, default='tresnet_l')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--dataset_type', type=str, default='MS-COCO')
    parser.add_argument('--th', type=float, default=None)

    # parsing args
    args = parse_args(parser)

    # Load image

    # feed into infer function
    state = torch.load(args.model_path, map_location='cpu')
    args.num_classes = state['num_classes']
    model = create_model(args).cuda()
    model.load_state_dict(state['model'], strict=True)
    model.eval()

    classes_list = np.array(list(state['idx_to_class'].values()))

    im = Image.open(args.pic_path)
    im_resize = im.resize((args.input_size, args.input_size))
    np_img = np.array(im_resize, dtype=np.uint8)
    np_imgs = np.stack([np_img, np_img])
    print(np_imgs.shape)
    tensor_img = torch.from_numpy(np_imgs).permute(0, 3, 1, 2).float() / 255.0  # HWC to CHW

    tensor_batch = tensor_img.cuda()

    probs, labels, labels_probs = infer_batch(model, classes_list, inputs=tensor_batch)
    print(probs)
    print(labels)
    print(labels_probs)