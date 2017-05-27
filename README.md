# Classification and Learning for Character Recognition

## Abstract
This report describes the details of building a classifier for a character set containing three letters: ’S’, ’T’ and ’V’.
Classification features are extracted from the magnitude spectrum of each individual letter, obtained after applying a
Fourier Transform. The classifier is build using the k-Nearest Neighbour (kNN) algorithm and it is tested using an
additional set of characters that include corner cases. In contrast to kNN, a Decision Tree classifier has been trained and
tested on the same set of characters. The report also contains a discussion on the differences between the two classifiers,
as well as advantages and disadvantages of each classification method.

## Fourier Domain Analysis
Any periodic function can be represented as a sum of sines and cosines, each multiplied by a corresponding coefficient.
Images can be represented as functions with two input variables, the output of the function being the gray-scale value of
a certain pixel. Since pixels are discrete values, we shall apply Discrete Fourier Transform [2]. The Fourier Transform
basically finds the coefficients of the sines and cosines. In order to apply this transform to an image, the following formula
needs to be used:

![1](https://cloud.githubusercontent.com/assets/16266257/26521040/e19c7794-42d6-11e7-93e1-122f5852d91c.PNG)

Intuitively, this formula computes the average brightness value for a certain couple of frequency values, using all
combinations of pairs of points (i.e. pixels).

Our data is comprised of images, which means that we are going to apply Fourier Transforms to move from the
spatial domain to the 2D frequency domain. The Fourier Transform, especially the magnitude spectrum, will offer more
information about the change of intensity along each dimension. Smooth areas in the image will result in low frequency,
whereas areas with rapidly changing intensity will result in high frequency. The output is another image with the
corresponding magnitude values at various frequency levels on both axis. For instance, when the frequency on both axis
is 0, i.e. u = 0 and v = 0, we get the average image gray-level. The further away we move from the center of this
image denoting the magnitude spectrum of the Fourier Space, the higher the frequency. Therefore, information about
the contrast of the original image is retained closer to the center, while information about finer details is enclosed in the
outer layers of the image. This allows us to control and manipulate the features that we want to extract from the image,
as shall be seen below.

The images in the Fourier Domain also have conjugate symmetry. To be more precise, each point has another
corresponding point with the same value across the symmetry point, i.e. the centre of the image. By looking at Figure
1, we can see that the top-left quarter of the image is symmetric with the bottom-right one; the other two quarters are
symmetric as well. In a memory-consuming application, only half of each image could be used in order to save space.

![s_avg](https://cloud.githubusercontent.com/assets/16266257/26521095/f3d45d54-42d7-11e7-81ad-09c2f7e41e06.png) ![t_avg](https://cloud.githubusercontent.com/assets/16266257/26521102/051ca508-42d8-11e7-8d2b-a16477716f37.png) ![v_avg](https://cloud.githubusercontent.com/assets/16266257/26521107/187bf6b2-42d8-11e7-84b8-915e224acc06.png)
