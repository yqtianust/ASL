import matplotlib
import matplotlib.pylab as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as panda
from matplotlib.patches import Polygon

def main():
    data = panda.read_csv("./summary_mlccoco_acc_top1gt_all_probs.csv")

    # x = data['real_acc'].to_numpy()
    overall_acc = data['acc']
    overall_acc = np.array([float(v.strip('%')) for v in overall_acc]) / 100.0

    MR1_acc = data['MR1_acc'].to_numpy()
    MR1_acc = np.array([float(v.strip('%')) for v in MR1_acc]) / 100.0
    non_MR1_acc = data['non_MR1_acc'].to_numpy()
    non_MR1_acc = np.array([float(v.strip('%')) for v in non_MR1_acc]) / 100.0

    MR2_acc = data['MR2_acc'].to_numpy()
    MR2_acc = np.array([float(v.strip('%')) for v in MR2_acc]) / 100.0
    non_MR2_acc = data['non_MR2_acc'].to_numpy()
    non_MR2_acc = np.array([float(v.strip('%')) for v in non_MR2_acc]) / 100.0

    MR12_acc = data['MR1&2_acc'].to_numpy()
    MR12_acc = np.array([float(v.strip('%')) for v in MR12_acc]) / 100.0
    non_MR12_acc = data['non_MR1&2_acc'].to_numpy()
    non_MR12_acc = np.array([float(v.strip('%')) for v in non_MR12_acc]) / 100.0


    # fig, ax = plt.subplots(figsize=(8, 6))
    plt.rcParams.update({'font.size': 24})
    from matplotlib.figure import figaspect
    w, h = figaspect(0.50)
    # fig, ax = plt.subplots(figsize=(w * 1.6, h * 1.6))
    fig, ax = plt.subplots(1,3,figsize=(w * 1.6, h * 1.6))

    arr = [overall_acc, MR1_acc, non_MR1_acc,
           overall_acc, MR2_acc, non_MR2_acc,
           overall_acc, MR12_acc, non_MR12_acc]
    # positions = [0, 2, 4]
    # label = ["Overall", "Violate", "Not Violate",
    #          "Overall", "Violate", "Not Violate",
    #          "Overall", "Violate", "Not Violate",]
    # color = ['lightpink', 'skyblue', 'lightgrey'], color = color[j]
    label = ["O", "U", "R"]
    marker = ["x", 'o', '+']
    positions = range(1, 7, 1)[0:3]
    for i in range(0, 3):
        for j in range(0,3):
            ax[j].scatter(["O", "U", "R"], [arr[3*j][i], arr[3*j+1][i], arr[3*j+2][i]], s = 200, marker=marker[i], linewidths=5)

    # ax[0].scatter(["O", "O", "O"], arr[0])
    # ax[0].scatter(["U", "U", "U"], arr[1])
    # ax[0].scatter(["R", "R", "R"], arr[2])
    # bp0 = ax[0].boxplot(arr[0:3], labels = label, positions=positions, widths=0.5)
    # bp1 = ax[1].boxplot(arr[3:6], labels = label, positions=positions, widths=0.5)
    # bp2 = ax[2].boxplot(arr[6:9], labels = label, positions=positions, widths=0.5)
    #
    # color = ['lightpink', 'skyblue', 'lightgrey']
    # bp = [bp0, bp1, bp2]
    # for k in range(3):
    #
    #     for i in range(3):
    #         box = bp[k]['boxes'][i]
    #         boxX = []
    #         boxY = []
    #         for j in range(5):
    #             boxX.append(box.get_xdata()[j])
    #             boxY.append(box.get_ydata()[j])
    #         box_coords = np.column_stack([boxX, boxY])
    #         # Alternate between Dark Khaki and Royal Blue
    #         ax[k].add_patch(Polygon(box_coords, facecolor=color[k]))

    # ax.boxplot(MR1_acc, loc=0)
    # ax.boxplot(non_MR1_acc, loc=1)
    # ax.boxplot(MR2_acc)
    # ax.boxplot(non_MR2_acc)
    # ax.boxplot(MR12_acc)
    # ax.boxplot(non_MR12_acc)


    # print(ax.get_xlim())
    # print(ax.get_ylim())
    print(fig.get_size_inches())

    # ax.set_xlim(0.69, 0.84)
    # ax[0].set_ylim(0.15, 0.95)
    # ax[1].set_ylim(0.15, 0.95)
    # ax[2].set_ylim(0.15, 0.95)

    ax[0].set_title("MR1")
    ax[1].set_title("MR2")
    ax[2].set_title("MR-1&2")

    # ax[0].tick_params(axis='x', rotation=45)
    # ax[1].tick_params(axis='x', rotation=45)
    # ax[2].tick_params(axis='x', rotation=45)

    ax[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax[2].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    # ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    # ax.set_title("The Relationship between Top-1 Accuracy\n and The Ratio of Inputs Violating MR")
    # ax.legend()
    ax[0].set_ylabel("Top-1 Accuracy")

    plt.suptitle("O: Original; U: Unreliable; R: Reliable",x=0.55, y=0.07, fontsize = 24)
    # ax[0].set_xlabel("O: Original;", horizontalalignment='left')
    ax[1].set_xlabel(" ")
    # ax[2].set_xlabel("R: Reliable")
    # fig.tight_layout()

    # ax.set_xlabel("Top-1 Accuracy")
    # fig.text(0.5, 0.0, 'Something', )
    plt.show()
    fig.savefig("./mlc_acc_vio_vs_nonVio_notitle3.pdf")

    from scipy.stats import mannwhitneyu
    print(mannwhitneyu(overall_acc, MR1_acc, alternative="greater"))
    print(mannwhitneyu(overall_acc, non_MR1_acc, alternative="less"))
    print(mannwhitneyu(MR1_acc, non_MR1_acc, alternative="less"))
    print(mannwhitneyu(overall_acc, MR2_acc, alternative="greater"))
    print(mannwhitneyu(overall_acc, non_MR2_acc, alternative="less"))
    print(mannwhitneyu(MR2_acc, non_MR2_acc, alternative="less"))
    print(mannwhitneyu(overall_acc, MR12_acc, alternative="greater"))
    print(mannwhitneyu(overall_acc, non_MR12_acc, alternative="less"))
    print(mannwhitneyu(MR12_acc, non_MR12_acc, alternative="less"))

if __name__ == '__main__':
    main()