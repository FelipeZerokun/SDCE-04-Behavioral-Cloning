import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


steering_center = []
steering_left = []
steering_right = []
left_images_path = []
center_images_path = []
right_images_path = []
correction = 0.5 # this is a parameter to tune

driving_data = np.array(pd.read_csv('simulation_data/driving_log.csv'))
print('data imported successfully!')

for data in range(len(driving_data)):
    ### Center images
    center_path = 'simulation_data/IMG/' + driving_data[data][0].split('\\')[-1]
    center_images_path.append(center_path)
    ### Left Images
    left_path = 'simulation_data/IMG/' + driving_data[data][1].split('\\')[-1]
    left_images_path.append(left_path)  
    ### Right Images
    right_path = 'simulation_data/IMG/' + driving_data[data][2].split('\\')[-1]
    right_images_path.append(right_path)
    
    ### Now the steering values. I added a 0.2 correction for left and right steering values
    steering_center.append(driving_data[data][3])
    steering_left.append(driving_data[data][3] + correction)
    steering_right.append(driving_data[data][3] - correction)


print('a total of ', len(driving_data), ' images paths were added')
print('image path example: ', center_images_path[0])

name = (right_images_path[1000]).strip()
image = mpimg.imread(name)
flipped_image= cv2.flip(image, 1)
plt.imshow(image)
plt.imshow(flipped_image)

y_values = []
X_paths = []
X_values = []

y_values.extend(steering_center)
y_values.extend(steering_left)
y_values.extend(steering_right)

X_paths.extend(center_images_path)
X_paths.extend(left_images_path)
X_paths.extend(right_images_path)

print('Number of x values:', len(X_paths))
print('Number of y values:', len(y_values))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_paths, y_values, test_size=0.2, random_state=20)

print('Training set len: ', len(X_train))

# Here I will create a Generator to handle the image processing
def generator(X_samples,y_samples, batch_size=32):
    num_samples = len(X_samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = X_samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in range(len(batch_samples)):
                name = (X_samples[batch_sample]).strip()
                #image = cv2.imread(name)
                image = mpimg.imread(name)
                angle = y_samples[batch_sample]
                augmentated_image = cv2.flip(image, 1)
                augmentated_angle = angle*(-1)
                images.append(image)
                images.append(augmentated_image)
                angles.append(angle)
                angles.append(augmentated_angle)
     
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train)
 
# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(X_train, y_train, batch_size=batch_size)
validation_generator = generator(X_test, y_test, batch_size=batch_size)

### NVidia model
import math
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Convolution2D

model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape = (160,320,3)))       # Normalizing the data
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3))) #Cropping the images
### This is the Nvidia autonomous vehicules architecture > 
# First, 5 Convolution layers
model.add(Convolution2D(24, (5,5), padding='same', activation='relu'))
model.add(Convolution2D(36, (5,5), padding='same', activation='relu'))
model.add(Convolution2D(48, (5,5), padding='same', activation='relu'))
model.add(Convolution2D(64, (3,3), padding='same', activation='relu'))
model.add(Convolution2D(64, (3,3), padding='same', activation='relu'))
# Four Fully connected layers
model.add(Flatten())
model.add(Dense(1164))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
            steps_per_epoch=math.ceil(len(X_train)*2/batch_size),
            validation_data=validation_generator,
            validation_steps=math.ceil(len(X_test)),
            epochs=2, verbose=1)
model.save('model.h5')
