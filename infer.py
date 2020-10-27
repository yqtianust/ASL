import torch
from src.helper_functions.helper_functions import parse_args
from src.models import create_model
import argparse
import numpy as np
from PIL import Image
import pickle

# import matplotlib

# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np

parser = argparse.ArgumentParser(description='ASL MS-COCO Inference on a single image')

parser.add_argument('--model_path', type=str, default='./pth_files/MS_COCO_TRresNet_L_448_86.6.pth')
parser.add_argument('--pic_path', type=str, default='./pics/000000000885.jpg')
parser.add_argument('--model_name', type=str, default='tresnet_l')
parser.add_argument('--input_size', type=int, default=448)
parser.add_argument('--dataset_type', type=str, default='MS-COCO')
parser.add_argument('--th', type=float, default=None)


def main():
    print('ASL Example Inference code on a single image')

    # parsing args
    args = parse_args(parser)

    # setup model
    print('creating and loading the model...')
    state = torch.load(args.model_path, map_location='cpu')
    args.num_classes = state['num_classes']
    model = create_model(args).cuda()
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    classes_list = np.array(list(state['idx_to_class'].values()))
    print('done\n')

    print('loading image and doing inference...')
    im = Image.open(args.pic_path)
    im_resize = im.resize((args.input_size, args.input_size))
    np_img = np.array(im_resize, dtype=np.uint8)
    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
    print(tensor_img.shape)
    tensor_batch = torch.unsqueeze(tensor_img, 0).cuda()
    print(tensor_batch.shape)
    output = torch.squeeze(torch.sigmoid(model(tensor_batch)))
    np_output = output.cpu().detach().numpy()

    with open("./np_output.pickle", 'wb') as handle:
        pickle.dump(np_output, handle)
    with open("./before_sigmod.pickle", 'wb') as handle:
        pickle.dump(model(tensor_batch).cpu().detach().numpy(), handle)

    detected_classes = classes_list[np_output > args.th]
    print('done\n')
    print(detected_classes)
    # print('showing image on screen...')
    # fig = plt.figure()
    # plt.imshow(im)
    # plt.axis('off')
    # plt.axis('tight')
    # # plt.rcParams["axes.titlesize"] = 10
    # plt.title("detected classes: {}".format(detected_classes))

    # plt.show()
    print('done\n')


if __name__ == '__main__':
    main()
