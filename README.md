# Classification and Learning for Character Recognition

## Abstract
This report describes the details of building a classifier for a character set containing three letters: ’S’, ’T’ and ’V’. Classification features are extracted from the magnitude spectrum of each individual letter, obtained after applying a Fourier Transform. The classifier is build using the k-Nearest Neighbour (kNN) algorithm and it is tested using an additional set of characters that include corner cases. In contrast to kNN, a Decision Tree classifier has been trained and tested on the same set of characters. The report also contains a discussion on the differences between the two classifiers, as well as advantages and disadvantages of each classification method.

## Fourier Domain Analysis
Any periodic function can be represented as a sum of sines and cosines, each multiplied by a corresponding coefficient. Images can be represented as functions with two input variables, the output of the function being the gray-scale value of a certain pixel. Since pixels are discrete values, we shall apply Discrete Fourier Transform [2]. The Fourier Transform basically finds the coefficients of the sines and cosines. In order to apply this transform to an image, the following formula needs to be used:

![1](https://cloud.githubusercontent.com/assets/16266257/26521040/e19c7794-42d6-11e7-93e1-122f5852d91c.PNG)

Intuitively, this formula computes the average brightness value for a certain couple of frequency values, using all combinations of pairs of points (i.e. pixels).

Our data is comprised of images, which means that we are going to apply Fourier Transforms to move from the spatial domain to the 2D frequency domain. The Fourier Transform, especially the magnitude spectrum, will offer more information about the change of intensity along each dimension. Smooth areas in the image will result in low frequency, whereas areas with rapidly changing intensity will result in high frequency. The output is another image with the corresponding magnitude values at various frequency levels on both axis. For instance, when the frequency on both axis is 0, i.e. u = 0 and v = 0, we get the average image gray-level. The further away we move from the center of this image denoting the magnitude spectrum of the Fourier Space, the higher the frequency. Therefore, information about the contrast of the original image is retained closer to the center, while information about finer details is enclosed in the outer layers of the image. This allows us to control and manipulate the features that we want to extract from the image, as shall be seen below.

The images in the Fourier Domain also have conjugate symmetry. To be more precise, each point has another corresponding point with the same value across the symmetry point, i.e. the centre of the image. By looking at Figure 1, we can see that the top-left quarter of the image is symmetric with the bottom-right one; the other two quarters are symmetric as well. In a memory-consuming application, only half of each image could be used in order to save space. 

![s_avg](https://cloud.githubusercontent.com/assets/16266257/26521095/f3d45d54-42d7-11e7-81ad-09c2f7e41e06.png "Logo Title Text 1")            |   ![t_avg](https://cloud.githubusercontent.com/assets/16266257/26521102/051ca508-42d8-11e7-8d2b-a16477716f37.png)  |  ![v_avg](https://cloud.githubusercontent.com/assets/16266257/26521107/187bf6b2-42d8-11e7-84b8-915e224acc06.png)
:-------------------------:|:-------------------------:|:-------------------------:

<p align="center">
Figure 1: Average logarithm magnitude spectrum of S, T and V character
</p>
## Feature Selection
Features describe the characteristics of the data. The combination of d features is represented as a d-dimensional column vector called a feature vector. The d-dimensional space defined by the feature vector is called the feature space. In order to classify characters, we had to create features that clearly separate the character classes. We have focused on the magnitude spectrum of a 2D Fourier transform. In order for magnitudes to be visible to the human eye, we took the logarithm of the magnitude spectrum. Without taking the log, all of the high values would be concentrated in the centre of the image, in the DC term. Figure 1 holds the average magnitude spectrum of each of the three characters.

From the figure above, you can notice that letter T has high magnitudes along the horizontal and vertical directions. This is due to a high rate of change of intensity in those two directions corresponding to the vertical and horizontal lines that form a letter T in the spatial domain. We have used this information when creating the features and decided that levels of high magnitudes are the ones that will clearly distinguish the classes. Following that analogy, we have created five filters: rotated rectangle, cross, plus, circle and sector, all of which can be seen in the Figure 2. As you can notice, all filters exclude the DC component, which is located in the center of the magnitude spectrum. The DC component represents the average brightness of an image in the spatial domain, which is not required in our model as the brightness of the image does not reveal anything about the displayed character. All filters are applied by extracting the pixels from the logarithm of the magnitude spectrum located in the filter region and then calculating the average value of those extracted pixels. We can therefore expect all T characters to return higher values than V and S characters when the plus filter is applied.

![rotated-box-filter](https://cloud.githubusercontent.com/assets/16266257/26521240/954b3a56-42db-11e7-8da1-d1e888f2d511.png) |  ![cross-filter](https://cloud.githubusercontent.com/assets/16266257/26521237/904bf662-42db-11e7-868b-c98e33a2546b.png) | 
![plus-filter](https://cloud.githubusercontent.com/assets/16266257/26521238/95232660-42db-11e7-8bee-66c68eb2d0b2.png) | ![ring-filter](https://cloud.githubusercontent.com/assets/16266257/26521239/953ed04a-42db-11e7-9296-1fa45a477a94.png) | ![sector-filter](https://cloud.githubusercontent.com/assets/16266257/26521249/d6124886-42db-11e7-9ed9-033960dac809.png) 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
 Rotated Box Filter | Cross Filter | Plus Filter | Ring Filter | Sector Filter
<p align="center">
Figure 2: Set of features selected from the original set
</p>
The choice of features is really important, as it influences the accuracy of the classification, the time needed for classification and the number of learning examples. Dimensionality reduction strives for compact representation of the properties of the data. The compact representation removes redundancy and can be obtained through feature selection and feature extraction. Feature selection involves selecting a subset of the existing features without a transformation. We have created a total of 5 features and used a heuristic strategy in selecting the 2 that make the clearest separation between the classes. We have first assumed that all features are independent. We have then selected a best single feature by using a significance test that involved observing the scatter plot matrix shown in Figure 3.

The scatter plot matrix represents every feature pair combination. On the diagonal, every feature is compared to itself and we used this information to determine a single best feature. We have chosen feature 1 (i.e. the cross filter), as it makes the clearest separation between one character (T in this case) and the other two. We chose the second feature by comparing all of the other features conditioned by the first one that was selected. It was hard to decide, as a lot of features were grouping the data in three distinguishable classes, but we have decided to select feature 2 (i.e the plus filter ) because, apart from one outlier, it separates the data in three well grouped classes.

![feature-grid](https://cloud.githubusercontent.com/assets/16266257/26521358/af1f1292-42de-11e7-91b9-b8f86cb44cd9.png)
<p align="center">
Figure 3: Comparison between pairs of features
</p>

## Feature Extraction
Feature Extraction refers to linear or non-linear transformations of the original variables to a lower dimensional feature space. It is useful when a feature space is too large, resulting in high computation times. A new set of features is built by mapping the features from the original set such that the transformed vector preserves most of the information given by the original set. An optimal mapping will be the one that results in no increase in the minimum probability error. Our feature space is two dimensional and does not suffer from the Curse of Dimensionality. We have experimented reducing the dimensionality but we were not successful in extracting a single feature that will preserve the clustering given by the two chosen features. This is due to the fact that all three characters have distinctive magnitude spectrums. Thus, a single feature can only distinguish one character from the other two. 

## Nearest-neighbour Classifier
The k-Nearest Neighbours algorithm (kNN) is a non-parametric lazy algorithm, meaning that it does not make any assumptions about the underlying model of the data[4]. For instance, it does not assume that the data is a Gaussian Mixture Model[1]. Also, the algorithm takes into account all of the training data during the testing phase. This can be costly in terms of how much memory the algorithm uses.

Whilst a very simple algorithm, kNN performs surprisingly well in practice. Below, in Figures 4, 5 and 6, you can see different decision boundaries when a different k is used. The kNN makes use of the majority rule. For each test data point, the algorithm works by taking the nearest k training data points, and looking at their labels. Each of these k training data points votes in favour of their own class. The class of the test data point is then set to the label with most votes. Usually, the distance measure is the euclidean distance and when we are performing binary classification, an odd k is chosen in order to avoid complications which arise in the case of two classes having an equal number of votes. As we have three classes, there is still a possibility of obtaining ties, but only in cases when the test point is close to where the boundaries meet.

![1nnclassigication_training](https://cloud.githubusercontent.com/assets/16266257/26521385/3f72e526-42df-11e7-9dd1-5e29def8c775.png) | ![3nnclassigication_training](https://cloud.githubusercontent.com/assets/16266257/26521386/3f8ad316-42df-11e7-8cbf-4e3aefcf8545.png) 
:-------------------------:|:-------------------------:
Figure 4: Nearest Neighbour Classification boundaries | Figure 5: 3-Nearest Neighbour Classification boundaries

![5nnclassigication_training](https://cloud.githubusercontent.com/assets/16266257/26521429/263e0328-42e0-11e7-9644-8840f484b1e9.png) | ![5nnclassigication_test](https://cloud.githubusercontent.com/assets/16266257/26521388/3f8fbc64-42df-11e7-8ab9-601c88f02cd0.png) 
:-------------------------:|:-------------------------:
Figure 6: 5-Nearest Neighbour Classification boundaries | Figure 7: 5-Nearest Neighbour Classification with test data

The graph is drawn by taking all the points on the meshgrid and determining their corresponding class. Each point is coloured according to its class. This creates a Voronoi diagram with 3 cells, one for each class. The training data points are denoted by circles, while the test data points are represented by stars. The average magnitude values computed initially from the training data have been normalized using the feature scaling method, bringing all of the values into the [0, 1] interval. Some of the test data falls out of this normalised interval, however, as can be seen in the graph.
