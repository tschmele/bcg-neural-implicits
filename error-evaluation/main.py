import numpy as np
import torch as th
from matplotlib import pyplot as plt
from models import SimpleModel

from sdfray.shapes import Sphere
from sdfray.util import *

rng = np.random.default_rng()

# ---
# load nn
# load sdf
# generate eval points
# for all points
#   run point through nn
#   run point through sdf
#   calculate (squared) difference
# plot all differences
# ---


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


def generate_err_histogram(error: np.array, save=False):
    plt.hist(error, histtype='bar')

    plt.title("Error Histogram")
    plt.xlabel("Error")
    plt.ylabel("Amount")

    if save:
        plt.savefig('histogram.png')
    plt.show()


def generate_err_bar_plot(error: np.array, save=False):
    error = np.sort(error)
    borders = np.array([])
    bars = np.array([])
    x_axis = np.array([])
    barrier = .0005
    step = .0005
    limit = .0055

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

    plt.title("Error Bar Plot")
    plt.xlabel("Error", loc='right', labelpad=-35.)
    plt.ylabel("Amount")

    plt.xticks(rotation='vertical')
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.15)

    if save:
        plt.savefig('bar-plot.png')
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nn = load_nn('NN.pth')
    sdf = load_sdf()
    pts = generate_points(100000)
    err = calculate_error(nn, sdf, pts)
    # generate_abs_err_histogram(err)
    generate_err_bar_plot(err)
