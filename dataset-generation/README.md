# Generating the Dataset for training

- - -

The goal is to generate 1M points and their distance to the nearest surface. These sets can then be used to train the Neural Implicit.

---

## V1 - SDFRay

Using [SDFRay](https://github.com/BenLand100/sdfray) we create the SDF of a sphere with radius 1. Following we generate a million points between ``[0., 0., 0.]`` and ``[1., 1., 1.]`` and calculate their distance to our sphere.

This pair of values gets stored in samples.txt and distances.txt which can be read into `` np.array``s for the following steps.  