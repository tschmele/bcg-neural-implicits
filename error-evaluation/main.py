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


def generate_plot(error: np.array):
    error.sort()
    plt.hist(error, histtype='bar')

    plt.title("Error Histogram")
    plt.xlabel("Error")
    plt.ylabel("Amount")

    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nn = load_nn('NN.pth')
    sdf = load_sdf()
    pts = generate_points(100000)
    err = calculate_error(nn, sdf, pts)
    generate_plot(err)

