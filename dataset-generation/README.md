# Generating the Dataset for training

- - -

The goal is to generate 1M points and their distance to the nearest surface. These sets can then be used to train the Neural Implicit.

---

## V1 - SDFRay and random samples

Using [SDFRay](https://github.com/BenLand100/sdfray) we create the SDF of a sphere with radius 1. Following we generate a million points between ``[0., 0., 0.]`` and ``[1., 1., 1.]`` and calculate their distance to our sphere.

This pair of values gets stored in samples.txt and distances.txt which can be read into `` np.array``s for the following steps.  

### Set generation

The parameter ``N`` defines the number of elements to be generated in a new set of traingdata. 

When generating a new set, the program starts by generating a ``numpy array`` with the shape ``(N, 3)`` which will be filled with random values between ``0.`` and ``1.``. These represent N points in 3D space, which are going to be used as input for the Neural Network.

Using a given SDF the program will go through all N points to calculate their distance from the surface of that SDF. In this example the SDF is a Sphere with ``radius = 1`` from the [SDFRay](https://github.com/BenLand100/sdfray) Library. These distances will provide the NN with the expected results during training.

### Storing and using sets

To use the set, the arrays with N points and N distances have to be stored in a way that will be accessible for training the NN and also preserve the set in a fixed state for repeated use and error checking.

For this purpose the program will create a folder ``./{N}-dataset/`` with N being the size of the set. This folder will contain the finished set. For the set with size ``N = 1000000`` the folder ``./1000000-dataset`` was created. 

Within this folder both the point and the distance array are stored as ``.txt`` files named ``samples.txt`` and ``distances.txt`` respectively. Afterwards the program stores another set of points and distances made up of the first ``int(N / 100)`` values of each array, which will be stored in ``samples_test.txt`` and ``distances_test.txt``. With that each training has a set of traingdata and testdata.

---

## Possible optimizations

To improve this method of generating datasets changes can be made primarily to the generation of the actual sets. Here it would improve the effect during training if the points were not purely random, but were created in proximity to the surface of the given SDF. For this the simplest approach would be to generate a larger set of points and then filter the N values which are closest to the surface. 

Alternatively it could be considered not generating purely random points and instead using a surface point as 'origin' and then generating a few points around this point by adding or subtracting small random values from all 3 coordinates. By repeating the second the option for various surface points it would provide a set large enough while providing only points in proximity to the given SDF. Additionally it provides the option to cluster points around specific features by selecting more 'origins' around these features.

A second option for optimization would be changing the storage of each dataset. But as of now it is not clear if changing the datatype would provide a significant improvement. Additionally these changes would only affect the process while storing a dataset and when reading it for the first time during training, but has no effect on the training itself.