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

![s18](https://cloud.githubusercontent.com/assets/16266257/26522589/e8536398-42fb-11e7-8d64-9d9d00a730b9.png) | ![s11](https://cloud.githubusercontent.com/assets/16266257/26522590/e8638d86-42fb-11e7-88af-85d1f0b89a94.png) | ![s12](https://cloud.githubusercontent.com/assets/16266257/26522592/e86b5426-42fb-11e7-9f2b-abf402fecd76.png) | ![s13](https://cloud.githubusercontent.com/assets/16266257/26522593/e86c4390-42fb-11e7-83d3-919b2cc58bc1.png) | ![s14](https://cloud.githubusercontent.com/assets/16266257/26522595/e86da44c-42fb-11e7-97ac-217661ea0149.png) | ![s15](https://cloud.githubusercontent.com/assets/16266257/26522591/e86ad4ec-42fb-11e7-88bd-7920e4feb36f.png) | ![s16-marked](https://cloud.githubusercontent.com/assets/16266257/26522596/e86fd83e-42fb-11e7-8f2a-611ffd21cf79.png) | ![s17](https://cloud.githubusercontent.com/assets/16266257/26522597/e8810bfe-42fb-11e7-9439-29136106be29.png)
:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
![t18](https://cloud.githubusercontent.com/assets/16266257/26522608/1b93fa7e-42fc-11e7-9592-c027e50bdb35.png) | ![t11](https://cloud.githubusercontent.com/assets/16266257/26522603/1b8e6834-42fc-11e7-92b4-3950af688791.png) | ![t12](https://cloud.githubusercontent.com/assets/16266257/26522605/1b902fac-42fc-11e7-824a-f02001edde2c.png) | ![t13](https://cloud.githubusercontent.com/assets/16266257/26522604/1b8f4b5a-42fc-11e7-8f9e-f8fac6747564.png) | ![t14](https://cloud.githubusercontent.com/assets/16266257/26522607/1b93713a-42fc-11e7-9480-84bfe2f2c452.png) | ![t15-marked](https://cloud.githubusercontent.com/assets/16266257/26522606/1b92ca0a-42fc-11e7-9ce3-aff03cb959e8.png) | ![t16](https://cloud.githubusercontent.com/assets/16266257/26522609/1ba8a960-42fc-11e7-8d73-e176616c2e49.png) | ![t17-marked](https://cloud.githubusercontent.com/assets/16266257/26522602/1b745f5c-42fc-11e7-960f-5eced62df8e5.png) 
![v18](https://cloud.githubusercontent.com/assets/16266257/26522616/3e3a9a74-42fc-11e7-9845-27dd12096bde.png) | ![v11](https://cloud.githubusercontent.com/assets/16266257/26522617/3e4dbece-42fc-11e7-8fdb-c254363f957d.png) | ![v12](https://cloud.githubusercontent.com/assets/16266257/26522618/3e5322b0-42fc-11e7-9492-d0e489247b9e.png) | ![v13-marked](https://cloud.githubusercontent.com/assets/16266257/26522622/3e567820-42fc-11e7-8144-46adc737daf6.png) | ![v14](https://cloud.githubusercontent.com/assets/16266257/26522620/3e55c100-42fc-11e7-82fb-1700569a4ee7.png) | ![v15](https://cloud.githubusercontent.com/assets/16266257/26522619/3e548dbc-42fc-11e7-9ef8-b4e387a18db5.png) | ![v16](https://cloud.githubusercontent.com/assets/16266257/26522621/3e560c00-42fc-11e7-964b-4cb1a5ac6375.png) | ![v17](https://cloud.githubusercontent.com/assets/16266257/26522623/3e668a76-42fc-11e7-8777-e257222c988c.png)
<p align="center">
Figure 8: Generated test data points
</p>

When designing our test data, we tried to create data points which would show the limits of our classifier. Some peculiar test data points which are analyzed below are marked in Figure 8. For instance, we have a included a small S, i.e. S16, which is rather sharp than curvy. Our classifier labels it correctly, but it places it in the middle of the graph, meaning it is not particularly confident about its decision. We can understand why by looking at the letter's magnitude spectrum included in Figure 9. This letter's magnitude spectrum is quite different from all of the average spectrums that we have computed. We can see the parts where V should be bright are also a bit brighter than the other regions. A similar example is letter V13. The most interesting ones are T15 and T17. The former is classified as a V, because it is tilted. Its magnitude spectrum overlays one of the cross filter's arms. The latter has a curvy top part, similar to how an S would look like, and therefore classifying it accordingly. If examined closely, the plus filter scores just bit higher than the cross filter. The magnitude spectrum in Figure 12 shows that, perhaps, the two arms of the plus filter cover the bright areas better than the arms of the cross filter. 

While some outliers are present, most of our test data has been classified correctly.

![s16_magnitude_spectrum](https://cloud.githubusercontent.com/assets/16266257/26522714/85ac35a0-42fe-11e7-8eb4-8214b3f1a107.png) | ![v13_magnitude_spectrum](https://cloud.githubusercontent.com/assets/16266257/26522717/85b4bff4-42fe-11e7-84fb-d693e3d6f494.png) | ![t15_magnitude_spectrum](https://cloud.githubusercontent.com/assets/16266257/26522715/85b43322-42fe-11e7-90d0-355fa1b144ad.png) | ![t17_magnitude_spectrum](https://cloud.githubusercontent.com/assets/16266257/26522716/85b46522-42fe-11e7-8fec-5f955c62b92f.png) 
:--------:|:--------:|:--------:|:--------:|
Figure 9: S16 magnitude spectrum | Figure 10: V13 magnitude spectrum | Figure 11: T15 magnitude spectrum | Figure 12: T17 magnitude spectrum

## Classifying A and B
An interesting exercise to do is to test the classifier which we have trained using the letters S, T and V on some other letters and see how well it performs, without tweaking or modifying anything. We have been given some test data for the letters A and B. The latter one is classified as an S. Its magnitude spectrum, shown in Figure 15, shows that the letter scores extremely high on the plus filter, its values being very close to 1. When taking away the horizontal, vertical, and oblique lines, we are left with a magnitude spectrum similar to the average magnitude spectrum of S. This makes sense because B contains lots of curves, just like the S.

![a-and-b](https://cloud.githubusercontent.com/assets/16266257/26522761/9753cb3c-42ff-11e7-818c-e3a2ab935820.png)
|:----:|
Figure 13: Classification of A and B

![a_magnitude_spectrum](https://cloud.githubusercontent.com/assets/16266257/26522760/97437f7a-42ff-11e7-9e69-0e6f93785de1.png) |  ![b_magnitude_spectrum](https://cloud.githubusercontent.com/assets/16266257/26522762/9755caea-42ff-11e7-9ab0-f0d6c66760dc.png)
:--------:|:--------:|
Figure 14: A magnitude spectrum | Figure 15: B magnitude spectrum

## Decision Trees
Decision trees [3] are common supervised classification data mining technique. The algorithm works by performing decisions at various levels of a tree, depending of which features have been chosen. It uses features which offer more information about the data first, i.e. features which classify the data better. The information gain can be determined by using an impurity measurement. The sklearn.tree.DecisionT reeClassif ier object has two impurity measurements, which are very similar: the Gini impurity and the Entropy. We decided to use the Gini impurity measurement, as it does not contain the computationally expensive logarithm and it is therefore slightly faster. The Gini impurity measures how often an element that was chosen randomly from the set of samples would be incorrectly labeled, provided that it was labeled randomly. The Gini impurity has the following formula, where c represents the number of classes:

![2](https://cloud.githubusercontent.com/assets/16266257/26522848/6fcefe04-4301-11e7-8a40-b1e378b4a9c2.PNG)

The decision tree is formed out of multiple nodes. The algorithm splits the data points in two sets at each step, until all the data points in a certain node are part of the same class or until there are no features left to use in order to make splits. Figure 16 shows that the algorithm does a fairly good job at determining the decision surfaces.

![decision-tree](https://cloud.githubusercontent.com/assets/16266257/26522763/975696f0-42ff-11e7-998d-676a0a12592f.png) | ![decision-tree-graphviz](https://cloud.githubusercontent.com/assets/16266257/26522764/975b2c2e-42ff-11e7-8a2f-5e9ff6d0c2ca.png)
:--------:|:--------:|
Figure 16: Decision surface of a decision tree using paired features | Figure 17: Diagram of the decision tree

Figure 17 exhibits a visual representation of the decision tree in a human understandable manner. The algorithm starts with all the samples in the root node. The first decision is based on the cross feature. The samples are split in such a way that the T training data points are to the left of the decision boundary (i.e. where the average brightness computed using the cross filter is smaller than 0.3416), while the other data points are to its right (i.e. where the values are greater than 0.3416). The left child, when the decision condition is T rue, contains only samples from the second class, so the algorithm has finished computing on this branch. For the right child, which contains all the sample points from class 1 and 3, the decision condition is based on the second feature (i.e. the plus feature in our case). At this point, the algorithm finishes computing because all of the leaves of the tree contain training data points from the same class.

Intuitively, decision trees emulate the way humans would try to set the decision boundaries. T’s score low on the cross feature, while the other two letters score relatively high. On the other hand, V ’s score low on the plus feature.

While decision trees do not tend to be as accurate as other approaches and tend to overfit the data if too many features are used, the technique has plenty of advantages. Being a white box model, it allows for a graphical representation of the tree, making it easy to look into why the algorithm took some decisions and gain more insight into the data. The Gini impurity allows for statistical tests to be applied in order to determine which features are more relevant in determining the best classification. A disadvantage is that the results it produces are not consistent - small variations in the training data can lead to a completely different tree. To mitigate the disadvantages, a small number of features is recommended to be chosen. In fact, the decision tree confirmed our assumption that the cross and the plus features are the best, by selecting those two features as the best ones when all five features are fed into the algorithm. The final decision tree was remarkably similar to the one which makes use of only two features.

Comparing the Decision Trees classifier with the k-Nearest Neighbours classifier, we can see that the results are quite similar, even if the boundaries generated by the decision tree are less precise. The mathematics behind the decision tree classifier is more complicated, but it offers another way of visualizing the data. Decision Trees are not affected by the outliers, whereas when using kNN, all points could potentially have an equal ”weighting” in building the decision boundary. In practice, the naive kNN is faster than naive decision trees. Assuming the decision tree is balanced, we have the following time complexities[5]:
* naive kNN: O(1) [training time] + O(NK) [query time] = O(NK)
* naive decision tree: O(N2 ∗ K ∗ log(N)) [training time] + O(log(N)) [query time] = O(N2 ∗ K)

## Conclusion
This report explored how Fourier Space Analysis can be used in letter recognition. It first gave a brief explanation of what the Fourier Domain is and how to apply the Fourier Transform in order to move to the frequency domain. Then, we discussed how to select and extract features from the Fourier Space to get the best possible classification. Afterwards, we examined two supervised classification algorithms and tested them on test data we generated.

### References

[1] Christopher M. Bishop. Pattern Recognition and Machine Learning. Springer, 2006.

[2] Sanjeev R. Kulkarni. Lecture Notes for ELE201 Introduction to Electrical Signals and Systems. 2002.

[3] skikit-learn Decision Trees. May 2017. url: http://scikit-learn.org/stable/modules/tree.html

[4] skikit-learn Nearest Neighbours. May 2017. url: http://scikit-learn.org/stable/modules/neighbors.html

[5] Why is KNN much faster than decision tree? May 2017. url: http://stackoverflow.com/questions/15428282/why-is-knn-much-faster-than-decision-tree

