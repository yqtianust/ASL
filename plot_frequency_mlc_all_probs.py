import matplotlib
import matplotlib.pylab as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as panda




def main():
    input = np.load("./mlc_input_frequency_all_probs.npz", allow_pickle= True)

    overall = input["overall"]
    times_input = input["times_input"]
    times_input_ratio = np.zeros_like(times_input).astype(float)
    for i in range(0, 3):
        times_input_ratio[i, :] = np.divide(times_input[i, :], overall)

    print(times_input_ratio)
    # exit(0)

    plt.rcParams.update({'font.size': 24})
    from matplotlib.figure import figaspect
    w, h = figaspect(0.55)
    fig, ax = plt.subplots(figsize=(w * 1.6, h * 1.6))

    print(ax.get_xlim())
    print(ax.get_ylim())
    print(fig.get_size_inches())

    w = 0.25
    x = np.array([i for i in range(1, 4, 1)])
    color= [ 'lightpink', 'skyblue', 'lightgrey']
    rect1 =ax.bar(x - w, times_input_ratio[:, 0], width=w, hatch='+',color=color[0], align='center', label='MR-1', alpha=.99)
    rect2 =ax.bar(x, times_input_ratio[:, 1], width=w, hatch='/',color=color[1], align='center', label='MR-2', alpha=.99)
    rect3 =ax.bar(x + w, times_input_ratio[:, 2], width=w,color=color[2], align='center', label='MR-1&2', alpha=.99)

    # ax.set_xlim(0.69, 0.84)
    # ax.set_ylim(0.02, 0.26)
    xlabels = [1, 2, 3]
    plt.xticks(x, xlabels)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2%}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    # autolabel(rect1)
    # autolabel(rect2)
    # autolabel(rect3)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    # ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    # ax.set_title("The Relationship between Top-1 Accuracy\n and The Ratio of Inputs Violating MR")
    ax.legend()
    ax.set_ylabel("Percentage")
    ax.set_xlabel("Number of Models ")
    plt.yticks(np.arange(0, 0.85, 0.10))
    plt.show()
    fig.savefig("./mlc_frequency_input_all_probs.pdf")


def main2():
    input = np.load("./label_frequency_all_probs.npz", allow_pickle=True)

    overall_label = input["overall_label"]
    times_label = input["times_label"]
    times_label_ratio = np.zeros_like(times_label).astype(float)
    for i in range(0, 11):
        times_label_ratio[i, :] = np.divide(times_label[i, :], overall_label)

    print(times_label_ratio)

    plt.rcParams.update({'font.size': 24})
    from matplotlib.figure import figaspect
    w, h = figaspect(0.618)
    fig, ax = plt.subplots(figsize=(w * 1.6, h * 1.6))

    print(ax.get_xlim())
    print(ax.get_ylim())
    print(fig.get_size_inches())

    w = 0.25
    x = np.array([i for i in range(1, 12, 1)])
    color = ['lightpink', 'skyblue', 'lightgrey']
    rect1 = ax.bar(x - w, times_label_ratio[:, 0], width=w, hatch='+', color=color[0], align='center', label='MR-1')
    rect2 = ax.bar(x, times_label_ratio[:, 1], width=w, hatch='x', color=color[1], align='center', label='MR-2')
    rect3 = ax.bar(x + w, times_label_ratio[:, 2], width=w, color=color[2], align='center', label='MR-1&2')

    # ax.set_xlim(0.69, 0.84)
    # ax.set_ylim(0.02, 0.26)
    xlabels = [1,2,3,4,5,6,7,8,9,10,r'$\geq11$']
    plt.xticks(x, xlabels)

    # autolabel(rect1)
    # autolabel(rect2)
    # autolabel(rect3)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    # ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    # ax.set_title("The Relationship between Top-1 Accuracy\n and The Ratio of Inputs Violating MR")
    ax.legend()
    ax.set_ylabel("Percentage")
    ax.set_xlabel("The Number of Models ")
    plt.show()
    fig.savefig("./frequency_label_all_probs.pdf")


if __name__ == '__main__':
    main()
    # main2()