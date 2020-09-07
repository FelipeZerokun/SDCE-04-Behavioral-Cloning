# **Behavioral Cloning** 

---

**Behavioral Cloning Project**
### By Felipe Rojas

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./template_images/nvidia_cnn_architecture.png "Nvidia Architecture"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model_creation.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of 5 convolution neural network layer with 3x3 filter sizes and depths between 32 and 64 and activation 'relu' for the non-linearity. (model_creation.py lines 94-98). After the convolutions, I added 4 fully connected layers, beginning with a flatten (model_creation.py line 100) and the layers with a Dropout of 0.5 (model_creation.py lines 101-107).

Before the layers, I added two data preproccesing lines. First, I normalized the data by dividing all the pixels by 255, and to to have a mean of 0, I substracted 0.5 from all the values. All of this with a Lambda function (model_creation.py line 90).
Right after that, I cropped my images to help the model to identify the relevant features within the image. By only having the road on the image, and cropping things like the mountains or the car bonnet, the model may find more relevant features for a better driving pattern. (model_creation.py line 91).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 102, 104, 106). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model_creation.py line 53). After I obtained good results (low both on training loss and validation loss), I tested the model.h5 with the Udacity car Simulator to see if the car effectively stayed in the road.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, both counter-clockwise and clockwise, recovering from the left and right sides of the road.
As recommended, I did 2 full laps counter-clock wise, 2 full laps clock wise and different recordings on the car recovering from the side of the lanes back to the center.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try different architectures, going from small ones to bigger ones, to see which one gave me better results. At the beginning, I just added a couple of layers to see how the training went, and the tried different configurations fo the Nvidia architecture.

My first step was to use add some layers to experiments and see how could I reduce the loss values with different parameters. or if I needed better training data. 
After some exploration, I decided to try a convolution neural network model similar to the Nvidia Architecture in the paper "End to End Learning for Self-Driving Cars"  I thought this model might be appropriate because it was build for a similar task with good results.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set in a 0.8 / 0.2 ratio. 
Firstly, my model performed poorly in both the training and validation set, so I added more convolutions. After some testing, I used 5 convolution layers, as in the Nvidia architecture and added more epochs to combat underfitting. Then, I used the Nvidia architecture and tested again. This time I found that the training set had low loss, but the validation set was not so good. Here, I added the Dropout layers between the fully connected layers. 
Another approach I could have used is gathering more diverse kind of data, for example, more aggresive side lane to center recovering, or using the other track of the car simulator.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
