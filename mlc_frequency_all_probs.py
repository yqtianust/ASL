import numpy as np
import matplotlib
import matplotlib.pylab as plt
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon

models = ['L', 'XL', 'MLCCOCO']

def main():
    # fig1, ax3 = plt.subplots(figsize=[20, 10])
    # fig2, ax2 = plt.subplots(figsize=[20, 10])
    # fig3, ax1 = plt.subplots(figsize=[20, 10])

    count = 0

    # with open("./filename_label.txt", 'r') as f:
        # file_label = f.readlines()

    # file_label = [v.split("; ")[2] for v in file_label]

    # labels = list(set(file_label))
    acc_all = np.zeros([40504, 3])
    arr_violate_obj = np.zeros([40504, 3])
    arr_violate_bg = np.zeros([40504, 3])
    arr_violate_both = np.zeros([40504, 3])
    # label_freq_obj = np.zeros([1000, 18])
    # label_freq_bg = np.zeros([1000, 18])
    # label_freq_both = np.zeros([1000, 18])
    # label_acc = np.zeros([1000, 18])

    for i in range(0, len(models)):
    # for i in range(1, 2):
        model_name = models[i]
        data =np.load("./mlc_all_probs_analyze_distribution/{}.npz".format(model_name), allow_pickle=True)

        acc = data['acc'][:,0]
        # prob = data['prob']
        acc_all[:,i] = acc
        violate_obj = data['violate_obj'][:,0]
        arr_violate_obj[:,i] = violate_obj
        violate_bg =  data['violate_bg'][:,0]
        arr_violate_bg[:, i] = violate_bg
        violate_both = data['violate_both'][:,0]
        arr_violate_both[:, i] = violate_both

        # for j in range(0, 50000):
        #     label = file_label[j]
        #     index_label = labels.index(label)
        #     label_acc[index_label, i] += acc[j]
        #     label_freq_obj[index_label, i] += violate_obj[j]
        #     label_freq_bg[index_label, i] += violate_bg[j]
        #     label_freq_both[index_label, i] += violate_both[j]

    # for i in range(0, len(models)):
    #     x = np.divide(label_freq_obj[:, i], label_acc[:, i])
    #     for j in range(i + 1, len(models)):
    #         y = np.divide(label_freq_obj[:, j], label_acc[:, i])
    #         s, p = wilcoxon(x, y)
    #         if p >= 0.05:
    #             print("{} {} {}".format(models[i], models[j], p))


    print(np.sum(arr_violate_obj, axis=1).shape)
    times_input = []
    overall = [np.sum(np.sum(arr_violate_obj, axis=1) > 0), np.sum(np.sum(arr_violate_bg, axis=1) > 0),
               np.sum(np.sum(arr_violate_both, axis=1) > 0)]
    print(overall)
    for time in range(1, 4):
        times_input.append([np.sum(np.sum(arr_violate_obj, axis=1) == time ),
                            np.sum(np.sum(arr_violate_bg, axis=1) == time),
                            np.sum(np.sum(arr_violate_both, axis=1) == time)])
    # times_input.append([np.sum(np.sum(arr_violate_obj, axis=1) > 10),
    #                     np.sum(np.sum(arr_violate_bg, axis=1) > 10),
    #                     np.sum(np.sum(arr_violate_both, axis=1) > 10)])
    for ele in times_input:
        print(ele)
    print(np.divide([np.sum(np.sum(arr_violate_obj, axis=1) == 3 ),
                            np.sum(np.sum(arr_violate_bg, axis=1) == 3),
                            np.sum(np.sum(arr_violate_both, axis=1) == 3)], overall))

    np.savez("./mlc_input_frequency_all_probs.npz", overall=overall, times_input = times_input)


    # label_yes_obj = (label_freq_obj > 0).astype(int)
    # label_yes_bg = (label_freq_bg > 0).astype(int)
    # label_yes_both = (label_freq_both > 0).astype(int)
    # print(np.sum(label_freq_obj, axis=1).shape)
    # overall_label = [np.sum(np.sum(label_yes_obj, axis=1) > 0), np.sum(np.sum(label_yes_bg, axis=1) > 0),
    #            np.sum(np.sum(label_yes_both, axis=1) > 0)]
    #
    # times_label = []
    # for time in range(1, 11):
    #     times_label.append([np.sum(np.sum(label_yes_obj, axis=1) == time), np.sum(np.sum(label_yes_bg, axis=1) == time),
    #         np.sum(np.sum(label_yes_both, axis=1) == time)])
    # times_label.append([np.sum(np.sum(label_yes_obj, axis=1) > 10), np.sum(np.sum(label_yes_bg, axis=1) > 10),
    #                 np.sum(np.sum(label_yes_both, axis=1) > 10)])
    #
    # np.savez("./label_frequency_all_probs.npz", overall_label=overall_label, times_label=times_label)

    # for i in range(0, len(models)):
    #     # for j in range(0, 1000):
    #
    #     num1 = np.sum(label_freq_obj[:, i] > 0)
    #     num2 = np.sum(label_freq_bg[:, i] > 0)
    #     num3 = np.sum(label_freq_both[:, i] > 0)
    #     print("{} {} {} {}".format(models[i], num1, num2, num3))





if __name__ == '__main__':
    main()