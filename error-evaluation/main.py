import torch as th
from matplotlib import pyplot as plt

from models import SimpleModel
from sdfray.shapes import Sphere
from sdfray.util import *

rng = np.random.default_rng()


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


def generate_err_histogram(error: np.array, save=None):
    plt.hist(error, histtype='bar')

    plt.title("Surface Error")
    plt.xlabel("Error")
    plt.ylabel("Amount")

    if save is not None:
        plt.savefig(save)
    plt.show()


def generate_err_bar_plot(error: np.array, save=None):
    error = np.sort(error)
    borders = np.array([])
    bars = np.array([])
    x_axis = np.array([])
    barrier = .0005
    step = .00025
    limit = .0003

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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nn = load_nn('NN.pth')
    sdf = load_sdf()
    pts = generate_points(100000)
    err = calculate_error(nn, sdf, pts)
    # generate_abs_err_histogram(err)
    generate_err_bar_plot(err)
