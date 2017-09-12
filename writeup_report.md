**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/sliding_window_1.5.png
[image4]: ./examples/sliding_window_2.0.png
[image5]: ./examples/sliding_window_3.0.png
[image6]: ./examples/sliding_windows_output.png
[image7]: ./examples/heatmap_output.png
[image8]: ./examples/heatmap_threshold_output.png
[image9]: ./examples/labels_map.png
[image10]: ./examples/bboxes_output.png
[image11]: ./examples/pipeline_output.png

[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 6th code cell of the IPython notebook in "./Vehicle-Detection.ipynb".  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here are few images for `vehicle`(1-10) and `non-vehicle` (11-20)classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed a random image from vehicle class dataset and displayed it to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I switched a lot between color spaces YUV and YCrCb. Although, I noticed the test validation accuracy pretty much the same with these two.

For the orientations, pixels per cell and cell per blocks params, I used the default values mentioned in the course. I didn't notice much difference when i changed these params.

I also used the spatial binning and histogram color features and combined hog features together.

I finally made use of `StandardScaler` class to normalize the features.

The code for these steps are contained in code cells 4, 5 and 7 in the IPython notebook.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a LinearSVM classifier using the normalized image features.

During the training, I split the data into train and test sets with the test set size equal to 2% of the total dataset size.

I used the Linear SVM model to fit the train data set and then validated the accuracy against the test dataset.

The accuracy was above 98%.

The code for this step is contained in the 8th code cell.  

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to split the sliding window search into 3 scales.

First, i applied sliding window with a scale value of 1.5 between vertical pixel postions 400-500. I came up with this value mostly after few trials. This scale is meant to detect cars at farther distance.

![alt text][image3]

Second, I applied sliding window with a scale of 2.0 between y positions 400 - 540. This sliding window is meant to detect near by cars(few meters).

![alt text][image4]

Lastly, I applied sliding window with a scale of 3.0 to detect cars that very close(few feet).

![alt text][image5]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YUV color spaced HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image6]

I primarily spent most of the time in figuring out proper vertical positions and scale values to accurately detect cars at different scales and positions.

Rest of code mostly came from class lecture note snippets.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is test and the corresponding heatmap:

![alt text][image7]

I then performed a threshold operation to filter out outliers or false positives.

Here is the output after applying threshold.

![alt text][image8]

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![alt text][image9]

Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image10]

### Here is the pipeline results over test images:

![alt text][image11]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I felt the multi scale sliding approach could've been more robust as I noticed few disturbances in the project video output.

I should try out few other values for params like orientations, color spaces and pixels per cells to see if the classifier accuracy improves.

I should've augmented some data to introduce shadows and low light scenarios to better find the cars.
