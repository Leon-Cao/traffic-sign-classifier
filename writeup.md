# **Traffic Sign Recognition** 

## Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization_histogram.png "Visualization"
[image2]: ./examples/grayscale_samples.png "Grayscaling"
[image3]: ./examples/random_noise_samples.png "Random Noise"
[image4]: ./examples/random_image_transform.png "Random images transform histogram"
[image5]: ./german-traffic-signs/00000.png "Traffic Sign 1"
[image6]: ./german-traffic-signs/00001.png "Traffic Sign 2"
[image7]: ./german-traffic-signs/00002.png "Traffic Sign 3"
[image8]: ./german-traffic-signs/00003.png "Traffic Sign 4"
[image9]: ./german-traffic-signs/00004.png "Traffic Sign 5"
[image10]: ./german-traffic-signs/00005.png "Traffic Sign 6"
[image11]: ./german-traffic-signs/00006.png "Traffic Sign 7"
[image12]: ./german-traffic-signs/00007.png "Traffic Sign 8"
[image13]: ./german-traffic-signs/00008.png "Traffic Sign 9"
[image14]: ./german-traffic-signs/00009.png "Traffic Sign 10"
[image15]: ./examples/predict_samples.png "Predict sample"
[image16]: ./examples/softmax_of_top5.png "softmax"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Leon-Cao/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is `34799`
* The size of the validation set is `4410`
* The size of test set is `12630`
* The shape of a traffic sign image is `(32, 32, 3)`
* The number of unique classes/labels in the data set is `43`

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the traffic sign has no strong relationship with color.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because to reduce the data set and make mean to 0.

I decided to generate additional data because from the visualization histogram figure. The number of traffic signs are too small (less than 300) and some are bigger than 1000. Then after shuffle processing, the less number traffic sign maybe less selected. Then it affects the training accuracy. Another reason is for real world condition, the light, shadow, contrast, position, rotation and other situation can change the traffic sign picture and make it hard to be recognized.

To add more data to the the data set, I used the following techniques
 1. random shift picture to other place to remove some pixels of traffic signs.
 2. random rotaion, to simulation real world situation of some traffic signs were changed by outside force.
 3. random blur, to simulate real work situation or camera problem or environment effects. 
 4. random brightness, to simulate strong light.
 5. random Gamma, to simulate low light situation.
 6. random shear, to simulate other view angle insteand of front view.

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 

![alt text][image4]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5     	| 5x5 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 3x3	    | 3x3 stride valid padding, outputs 12x12x32. 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x32  				    |
| Fully connected		| 6x6x32 -> 1152.        						|
| Fully connected		| 1152 -->384.        							|
| RELU					|												|
| dropout				| ratio = 0.5									|
| Fully connected		| 384 -->43.        							|
| Softmax				| .        									 |

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used epoch = 120, batch size = 128, learning ratio = 0.0009, mu=0, sigma=0.1

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation accuracy of 0.972
* test set accuracy of 0.950

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  * `first I choose LeNet-5, but the validation accuracy can only achieve 0.92`
* What were some problems with the initial architecture?
  * `I add dropout function, the validation accuracy can achieve 0.96 but it is not stable.`
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  * `I tried remove the last full connection layer and full connection layer from 3 to 2`
  * `Also add the matrix number of tensor`
* Which parameters were tuned? How were they adjusted and why?
  * `Learning ratio was tuned. Tried 0.001, 0.0001, 0.0003 and 0.0009. Seems the 0.0009 is better. 0.0001 is too small and the validation accuracy is bad.`
  * `Learning Epoch was tuned. Seems the epoch is bigger, the result is better. Finally selected a middle value`
  * `Batch was tuned. Tried 80, 98, 128 and 200. the validation accuracy is shake too much and 98 and 128 are similiar. 200 is bad.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  * `Dropout layer help create a more better model.`

If a well known architecture was chosen:
* What architecture was chosen?
  * `CNN and LeNet-5`
* Why did you believe it would be relevant to the traffic sign application?
  * `From the example, the CNN or LeNet-5 learn features from simple to complex. So, it is good for traffic signs.`
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
  * `Validation accuracy is 0.972 and test accuracy is 0.95.`
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 10 German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10] ![alt text][image11] ![alt text][image12] ![alt text][image13] ![alt text][image14]

The main diffent is the image size of each picture. So, I change the image size by `cv2.resize()` function. Then do the test.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image15]

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 26th cell of the Ipython notebook. For the probabilities, please refer below image

![alt text][image16]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I am so sorry that my second baby was born on Jan-12. So, I have no time to do the optional task.
