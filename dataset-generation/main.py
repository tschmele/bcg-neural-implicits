import numpy as np

# uses the sdfray library by BenLand100
# https://github.com/BenLand100/sdfray
from sdfray.light import DistantLight, PointLight, AmbientLight
from sdfray.scene import Scene
from sdfray.shapes import Sphere

rng = np.random.default_rng()
sphere = Sphere()

N = 1000000


def render_sphere(path: str):
    lights = [
        PointLight([10, 0, 0], [2, 2.5, -2]),
        PointLight([0, 10, 0], [-2, 2.5, 0]),
        DistantLight([0, 0, 1.5], [2, 2.5, 2]),
        AmbientLight([0.1, 0.1, 0.1])
    ]

    Scene(sphere, lights).cpu_render().save(path)


def generate_random_samples(sdf, n: int) -> (np.array, np.array):
    samples = rng.random((n, 3))
    distances = np.array([])
    for pts in samples:
        distances = np.append(distances, [sdf.fn(pts)])

    distances = distances.reshape((n, 1))
    return samples, distances


def save_dataset_to_txt(samples: np.array, distances: np.array, out_s: str, out_d: str):
    with open(out_s, 'w') as sam:
        for i in samples:
            np.savetxt(sam, i)
    print('Samples exported to file')
    with open(out_d, 'w') as dis:
        for i in distances:
            np.savetxt(dis, i)
    print('Distances exported to file')


if __name__ == '__main__':
    render_sphere('sphere.png')

    # currently set to 1M samples !
    # takes some time
    s, d = generate_random_samples(sphere, N)
    save_dataset_to_txt(s, d, 'samples.txt', 'distances.txt')
