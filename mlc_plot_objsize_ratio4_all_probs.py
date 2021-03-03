import numpy as np
import matplotlib
import matplotlib.pylab as plt
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
import matplotlib.ticker as mtick
import math
# models =  ['xception', 'vgg16', 'vgg19', 'resnet50',
#            'resnet101', 'resnet152', 'resnet50_v2', 'resnet101_v2',
#            'resnet152_v2',  'inception_v3', 'inception_resnet_v2' ,'mobilenet',
#            'mobilenet_v2', 'densenet121', 'densenet169', 'densenet201',
#            'nasnet_mobile', 'nasnet_large']

models =  ['resnet101' ,'mobilenet', 'nasnet_large', 'vgg16']

def main():
    plt.rcParams.update({'font.size': 24})
    from matplotlib.figure import figaspect
    w, h = figaspect(0.55)
    fig, ax = plt.subplots(figsize=(w * 1.6, h * 1.6))


    for j in range(2, 3):
    # for i in range(1, 2):
        model_name = models[j]
        data =np.load("./all_probs_analyze_distribution/{}.npz".format(model_name), allow_pickle=True)
        fig, ax = plt.subplots(figsize=(w * 1.6, h * 1.6))
        # acc = data['acc']
        # prob = data['prob']
        violate_obj = data['violate_obj']
        violate_bg =  data['violate_bg']
        violate_both = data['violate_both']

        obj_size = np.zeros([21])
        all_obj = np.zeros([21])
        bg_size = np.zeros([21])
        all_bg = np.zeros([21])
        both_size = np.zeros([21])
        all_both = np.zeros([21])

        sizes = np.load("obj_size_ratio.npy")
        for i in range(0, 50000):
            size = sizes[i]
            id = math.floor(size / 0.05)
            all_obj[id] += 1
            all_bg[id] += 1
            all_both[id] +=1
            if violate_obj[i]:
                obj_size[id] += 1
            if violate_bg[i]:
                bg_size[id] += 1
            if violate_both[i]:
                both_size[id] += 1


        x = np.array(range(0, 21)) * 0.05
        # ax[j].plot(x, np.divide(obj_size, all_obj.astype(float)), '+-',lw=3, ms=14,color = '#856060', label="MR-1")
        # ax[j].plot(x, np.divide(bg_size, all_bg.astype(float)), 'x-',lw=3, ms=14,color = '#64705c',label="MR-2")
        # ax[j].plot(x, np.divide(both_size, all_both.astype(float)),'d-',lw=3, ms=14,color = '#9c8563', label="MR-1&2")
        ax.plot(x, np.divide(obj_size, all_obj.astype(float)), '+-', lw=3, ms=14, color='#856060', label="MR-1")
        ax.plot(x, np.divide(bg_size, all_bg.astype(float)), 'x-', lw=3, ms=14, color='#64705c', label="MR-2")
        ax.plot(x, np.divide(both_size, all_both.astype(float)), 'd-', lw=3, ms=14, color='#9c8563', label="MR-1&2")


        ax.legend()
        ax.set_ylim(-0.02, 0.6)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax.xaxis.set_major_locator(mtick.MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.05))
        ax.set_ylabel("Percentage of Unreliable Inferences")
        ax.set_xlabel("Ratio of Target Object Size")
        plt.show()
        fig.savefig("./obj_size_{}_all_probs.PDF".format(model_name))


if __name__ == '__main__':
    main()