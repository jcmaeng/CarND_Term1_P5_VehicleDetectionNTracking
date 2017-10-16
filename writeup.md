# **Vehicle Detection Project**


The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/f1_car_not_car.jpg
[image2]: ./output_images/f2_car_hog_img.png
[image3]: ./output_images/f3_hog_subsampling.jpg
[image4]: ./output_images/f4_heatmap.jpg
[image5]: ./output_images/f5_remove_false_positives.jpg
[image6]: ./output_images/f6_spatial_hist_features.png
[video1]: ./out_project_p5.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The main procedure is contained in the function named `detect_vehicles()` (from line 25 to 184 of `main.py`). 
The code for extracting HOG features is contained in the first member function of `CarFinder` class in `car_finder.py`.  This function is `get_hog_features()` which returns HOG features of input images.
Before extracting HOG features, I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters to extract appropriate HOG features from image.
This table shows the parameters which I had finally chosen and tested values.

| Parameter Name| Value     | Tested values|
|:--------------------|------------|:--------------|
|orientations        | 11         |  9, 11, 12, 16  |
|pix_per_cell         | (16, 16)    | 16, 32 |
|cell_per_block     |  (2, 2)        | 1, 2 |
|hog_channel      |  ALL         | 0, 1, 2, or 'ALL' |

I chose these parameter values by lots of experiments. I run this algorithm with each parameter values and check the result image repeatedly.


#### 3 . Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVC using extracted features which are HOG features for all channels, spatial and color features. The code for this step is contained in the `detect_vehicles()` (from line 78 to 111). After extracting features to need from those images, then stacking whole feature data for training with `np.vstack()` and make label data with `np.hstack()`. I normalized with `sklearn.preprocessing.StandardScaler` and split up data into randomized training and test sets. (the ratio of test set is 20%)

The classifier is trained with parameter C=0.0001, where this value was selected based on the accuracies of train and test set. After some training, the training accuracy was always high, so the C was lowered because it meant overfitting. The final accuracy obtained on the test set was 98.41%. The classifier and the scaler were saved using pickle, so they can be reused whenever the camera images coming to process.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I tried to use the sliding window search directly but it did not show good result. So I applied HOG subsampling window search method from the lesson 35. :) After dozens and hundreds of test, I decided to apply scales of 1.0 and 1.5. Too many scales were not useful to me. As you can see the code in the line 130 of `main.py`, `find_cars()` function of `CarFinder` was invoked for each scale and obtained position list of cars.  

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

Heatmap from HOG sub-sampling image
![alt text][image4]

Image after removing false positives
![alt text][image5]

Spatial and color histogram features
![alt text][image6]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/out_project_p5.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  
I accumulated 5 images before applying threshold to take better results.

I didn't take capture images from video. I will update some more example images.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I spent so many times to tune the parameters. And I failed to eliminate false positives especially on the guardrail and shadows. I think that the brightness adjustment may be useful to that case. I will try that.

