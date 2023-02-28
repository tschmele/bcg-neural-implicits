# Evaluating the trained Neural Networks

- - -

The goal is to provide a way to analyse and evaluate the accuracy of a trained Neural Network.

---

## V1 - Absolute error analysis

After reading in both a Neural Network and the corresponding SDF, a set of random points for evaluation is generated. The absolute error is calculated by subtracting the distance to the nearest surface calculated by the SDF from the estimation by the NN and taking the absolut value of the result.

These absolute errors are then plotted using ``pyplot`` for manual evaluation.

### Error calculation

To calculate any error, the program starts by generating a set of random values between ``1.`` and ``0.`` in the shape ``(test_n, 3)``. This provides ``test_n`` different points in 3D space. 

Using these points it calculates both the true distance of each point by using the given SDF and the then the approximation calculated by the Neural Network. Afterwards it calculates the absolute difference between these values and stores them in an array. This error array is returned and can be used in subsequent steps.

### Plotting

The error array is turned into plots using the ``pyplot`` library from ``matplotlib``. For this the program has 2 separate methods.

One simply generates an automatic histogram from the given error array. This is both the easiest method and the most used method. Pyplot automatically selects bins based on the input array and then creates the histogram. After adding a title and labels, the finished plot is stored in a folder ``./{N}-plots/`` with N indicating the NN that is being analysed. 

For the other method the program provides a start-value, a step-size and a limit, which are then used to generate a bar plot emulating a custom histogram. For this the error array is first sorted. Then each value is compared to the current ``barrier`` with the first being the provided start-value. The index of the first value ``e`` with ``e >= barrier`` is stored and the barrier is increased by ``step-size``. This goes on until ``barrier`` becomes equal to or bigger than the given ``limit``.

By knowing the indices at which a given bin ends, the bar sizes can be calculated by taking the difference between the ends of the previous and the current bin. These bins and their corresponding barriers are then used to create a bar plot, which is labeled and stored in the same way as the histograms.

---

## V2 - Epoch comparison

To automate the generation of plots for every epoch of a given NN the previous program was modified and extended. Additionally it stores a plot of the average error per epoch.

### Plotting

To automate the plotting for 100 different NNs, the method takes ``path`` as input which points to a folder containing each NN as input instead of preloading one NN as in the previous version. It then loops over all 100 epochs while repeating the same steps. First it loads the current NN ``{path}SimpleModel_epoch_{i}.pth`` with ``i`` being the number of the current iteration and provides it with the corresponding SDF to the same error calculation method as in V1. 

The error array is then turned into a histogram as in V1 and stored. Additionally, all values with ``e <= .01`` are used to generate a second histogram. This step ensures a plot for each epoch with the same scale on the x-axis for easier comparison. The value ``.01`` was chosen after seeing that the majority of all values lies below or around that value in all previously analysed NNs. In the end the folder for these plots contains two histograms per epoch each with their epoch number in both title and filename.

### Average error

In addition to creating two plots for each epoch, the program calculates the average of each error array and stores these averages in order. These averages can be used to more accurately track the improvement over multiple epochs than visually comparing each plot. After finishing up the last epoch, all these average values are then used to generate a line plot which is saved alongside the other plots and a textfile containing the precise average error values, in case they will be needed in further analysis.

---

## Possible optimization

As of now the program mainly generates plots. The actual analysis and evaluation happens manually. Therefore, the most direct way of improvement is to add functionality which can use the error arrays to evaluate the performance of each NN automatically.

Additionally, the program as of now only processes absolute errors and their average. To increase the evaluation options, the program can be extended with different methods for calculating error such as MSE(mean squared error) or RMSE(rooted mean squared error). Also the program could extend into different classification metrics such as Accuracy, Precision, Recall and F1-Score [Performance Metrics in Machine Learning](https://neptune.ai/blog/performance-metrics-in-machine-learning-complete-guide).

Which of these options are suitable to evaluate the given use case can be determined the easiest after running a few tests with each.