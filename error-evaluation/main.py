import os
import time

import numpy as np
import torch as th
from matplotlib import pyplot as plt

from models import SimpleModel
from sdfray.shapes import Sphere
from sdfray.util import *

rng = np.random.default_rng()

path = '../neurales-netzwerk/'
N = '128k'
test_n = 10000
dataset = f'{N}-dataset/'
plt_dir = f'{N}-plots/'


def load_nn(path: str):
    model = SimpleModel()
    model.load_state_dict(th.load(path))
    model.eval()
    return model


def load_sdf():
    return Sphere()


def generate_points(n: int) -> np.array:
    return rng.random((n, 3))


def calculate_error(nn, sdf, pts: np.array) -> np.array:
    error = np.array([])

    for p in pts:
        tru = sdf.fn(p)
        app = nn(th.from_numpy(p))
        error = np.append(error, [np.abs(tru - app.item())])

    return error


def generate_err_histogram(error: np.array, title='Surface Error', save=None):
    plt.hist(error)

    plt.title(title)
    # plt.xlabel("Error")
    plt.ylabel("Amount")

    if save is not None:
        plt.savefig(save)
    plt.show()


def generate_err_bar_plot(error: np.array, barrier=.0005, step=.00025, limit=.003, save=None):
    error = np.sort(error)
    borders = np.array([])
    bars = np.array([])
    x_axis = np.array([])

    for idx, e in np.ndenumerate(error):
        if barrier < limit and e >= barrier:
            borders = np.append(borders, idx[0])
            x_axis = np.append(x_axis, np.round(barrier, 5))
            barrier += step

        if barrier >= limit:
            borders = np.append(borders, len(error))
            x_axis = np.append(x_axis, 'more')
            break

    prev = 0

    for b in borders:
        bars = np.append(bars, b - prev)
        prev = b

    plt.bar(x_axis, bars)

    plt.title("Surface Error")
    # plt.xlabel("Error", loc='right', labelpad=-35.)
    plt.ylabel("Amount")

    plt.xticks(rotation='vertical')
    # plt.margins(0.1)
    plt.subplots_adjust(bottom=0.15)

    if save is not None:
        plt.savefig(save)
    plt.show()


def epoch_comparison(path: str):
    sdf = load_sdf()
    pts = generate_points(test_n)
    avg_err = np.zeros((100, 1))
    if not os.path.isdir(f'{plt_dir}'):
        os.makedirs(f'{plt_dir}')

    for i in range(1, 101):
        print(f'=== epoch {i} in progress ===')
        t0 = time.time()

        nn = load_nn(f'{path}SimpleModel_epoch_{i}.pth')
        err = calculate_error(nn, sdf, pts)
        avg_err[i-1] = np.average(err)

        generate_err_histogram(err, title=f'Epoch {i} Surface Error', save=f'{plt_dir}hist_epoch_{i}.png')
        generate_err_histogram([x for x in err if x <= .01], title=f'Epoch {i} Surface Error', save=f'{plt_dir}hist_short_epoch_{i}.png')

        t1 = time.time()
        print(f'\n-> done ! {((t1 - t0) * 1000.0):.2f} ms')
        print(f'avg error : {avg_err[i-1]}')

    with open(f'{plt_dir}avg_err.txt', 'w') as txt:
        for i in avg_err:
            np.savetxt(txt, i)


def plot_avg_err(avg_err: np.array):
    plt.plot(avg_err)

    plt.title(f'dataset {N} , lr = .0001')
    plt.ylabel('Average Error')
    plt.xlabel('Epoch')

    plt.show()


def main():
    nn = load_nn('NN.pth')
    sdf = load_sdf()
    pts = generate_points(test_n)
    err = calculate_error(nn, sdf, pts)

    generate_err_histogram(err)
    generate_err_histogram([x for x in err if x <= 0.01])
    # generate_err_bar_plot(err)
    generate_err_bar_plot(err, barrier=.00025, step=.00025, limit=.01)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # main()
    # epoch_comparison(f'{path}{dataset}')
    avg_err = np.loadtxt(f'{plt_dir}avg_err.txt')
    plot_avg_err(avg_err)
