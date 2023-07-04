from scripts.utils import draw_range
import matplotlib.pyplot as plt
import numpy as np


def load_PR(p_file, r_file):
    precision = np.load(p_file)
    recall = np.load(r_file)
    return precision, recall


def get_loss_way(dataset, loss_ways, bit):
    p_files = []
    r_files = []
    for item in loss_ways:
        p_proto_file = f'./results/{dataset}/{item}/P_{bit}.npy'
        r_proto_file = f'./results/{dataset}/{item}/R_{bit}.npy'
        p_files.append(p_proto_file)
        r_files.append(r_proto_file)
    return p_files, r_files


def draw_curves(dataset, loss_ways, bit):
    markers = "DdsPvo*xH1234h"
    method2marker = {}
    i = 0
    for loss_way in loss_ways:
        method2marker[loss_way] = markers[i]
        i += 1

    p_files, r_files = get_loss_way(dataset, loss_ways, bit) # get all precision files and recall files
    pr_data = {}
    j = 0
    for loss_way in loss_ways:
        precision, recall = load_PR(p_files[j], r_files[j])
        pr_data[loss_way] = [precision, recall]
        j += 1

    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    for method in pr_data.keys():
        P, R = pr_data[method]
        plt.plot(R, P, linestyle="-", marker=method2marker[method], label=method)
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend()

    plt.subplot(132)
    for method in pr_data:
        P, R = pr_data[method]
        plt.plot(draw_range, R, linestyle="-", marker=method2marker[method], label=method)
    plt.xlim(0, max(draw_range))
    plt.grid(True)
    plt.xlabel('The number of retrieved samples')
    plt.ylabel('recall')
    plt.legend()

    plt.subplot(133)
    for method in pr_data:
        P, R = pr_data[method]
        plt.plot(draw_range, P, linestyle="-", marker=method2marker[method], label=method)
    plt.xlim(0, max(draw_range))
    plt.grid(True)
    plt.xlabel('The number of retrieved samples')
    plt.ylabel('precision')
    plt.legend()
    plt.savefig("pr.png")
    plt.show()