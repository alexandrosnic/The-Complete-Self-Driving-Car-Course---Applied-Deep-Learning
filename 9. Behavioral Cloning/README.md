# README

## Behavioral Cloning

![](./docs/Test.gif)

In this tutorial we are going to ultimately build an autonomous driving Neural Network model, trained from manual driving simulator, using the behavioral cloning technique.
I will use Google collab for the visual representation rather than Jupyter lab, since it offers free computational power.

The tutorial was part of the  **"Complete Self-Driving Car Course - Applied Deep Learning" course of Udemy.**

This tutorials will apply all of the concepts we dig into the previous tutorials:
- Deep Neural Networks
- Feature Extraction using Convolutional Neural Networks
- Continuous Regression

<img width="895" alt="whatwilluse" src="https://user-images.githubusercontent.com/34197007/80828103-0708a480-8be5-11ea-87b8-da21ab0f3207.PNG">

We will use a driving simulator built in Unity, provided for free by Udacity, that can be found here:

[https://github.com/udacity/self-driving-car-sim](https://github.com/udacity/self-driving-car-sim)

<img width="672" alt="simulator" src="https://user-images.githubusercontent.com/34197007/80828096-04a64a80-8be5-11ea-84d5-6dd71b64d7ae.PNG">

## I. Polynomial Regression - Background

While manually driving, we will continuously capture images that are going to be our training dataset and get the steering angle at each specific instance. Then these images will be fed into the Deep NN to help our model learn how to drive, by adjusting the steering angle, a method called **behavioral cloning**.


After we define the model, we are going to test its performance in a completely different testing track, where the car will drive autonomously.

For the implementation of Behavioral Cloning, we will make use of **Polynomial Regression** for the steering angle value. So far we used NN to classify our data. However with polynomial regression we are going to build a model to fit these data, and use it to predict the next most appropriate steering angle. To build the polynomial regression, we use Neural Networks to adjust the weights of the curve, in order to minimize the **mean square errors** and build a model that accurately fits the data.

<img width="768" alt="imagesNN" src="https://user-images.githubusercontent.com/34197007/80828134-0c65ef00-8be5-11ea-8dbd-35f1f9ed9172.PNG">


## II. Collect the data:

We will use the Udacity's simulator to train our model. For this, we select **Training Mode** and we do 3 laps clockwise, and 3 laps anti-clockwise, driving in the middle so that our model will be generalized and won't be biased towards one direction.

The data are generated from 3 cameras on the left, the middle and the right of the windshield, collecting data for the steering angle (in radians), speed, throttle and brake. Thus, a **csv** and a file containing **images** are created. Ultimately, each image feature (dataset) is associated with a label representing the **steering angle value**, from 1 (right) to -1 (left), making it a regression-based problem rather than classification.

![](./docs/Training.gif)

## III. Examine the data:

### 1. Clone the data:

We first **clone** the data we created from this github repository:

`!git clone https://github.com/alexandrosnic/The-Complete-Self-Driving-Car-Course---Applied-Deep-Learning`

### 2. Read the data:

Then we read the data:

`data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names = columns)`

and we get this datasheet:

<img width="813" alt="csvsample" src="https://user-images.githubusercontent.com/34197007/80828121-0a039500-8be5-11ea-8667-5ce957949d07.png">


### 3. Visualize the distribution:

We **visualize** the distribution of the steering angles in a histogram, just to get an idea, and to determine the steps we need to do in the preprocessing section. We also set a **threshold** of 200 samples in each bin to make the distribution more uniform:

```
num_bins = 25
samples_per_bin = 200
hist, bins = np.histogram(data['steering'], num_bins)
center = (bins[:-1]+ bins[1:]) * 0.5  # To normalize the data
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
```

we get this histogram:

<img width="285" alt="histogram" src="https://user-images.githubusercontent.com/34197007/80828132-0bcd5880-8be5-11ea-84cb-11c6dfaad6a5.PNG">

and we notice that most of the values belong in the bar around the angle 0. In order to make a better use of the data, we will exclude the data that exceed the threshold, after we first **shuffle** them to make sure we still have data from every part of the track in our final dataset. We get the updated histogram which seems to be much more uniform:

<img width="286" alt="histogram2" src="https://user-images.githubusercontent.com/34197007/80828133-0bcd5880-8be5-11ea-8268-978fc08c3ae4.PNG">

## IV. Create the training and validation datasets:

We need to split our images and steering values from the initial dataset into a **training** and **validation** set. 

To do so, we iterate through the entire initial dataset, we obtain the paths for the images and the values for steering, and we append them into **arrays**.

```
image_paths = np.asarray(image_path)
steerings = np.asarray(steering)
```
Then we **split** these two arrays into training and validation sets (of size 20%) by using the `train_test_split` sklearn tool :

`image_paths, steerings = load_img_steering(datadir + '/IMG', data)`

We **visualize** them as histograms to evaluate the splitting:

<img width="541" alt="trainvalidhisto" src="https://user-images.githubusercontent.com/34197007/80828101-0708a480-8be5-11ea-9d6c-5ab6b5a6be5a.PNG">

We notice that both the training and the validation set are **equally balanced around the center** which means that both sets are uniform, and thus can be fed to the model we will build.

## V. Preprocess the data:

### 1. Obtain the images:

We **obtain** the images by taking their path:
`mpimg.imread(img)`

We compare the original with the preprocessed image by **visualizing** them:

```
image = image_paths[100]
original_image = mpimg.imread(image)
preprocessed_image = img_preprocess(original_image)
```

<img width="797" alt="comparison" src="https://user-images.githubusercontent.com/34197007/80828117-096afe80-8be5-11ea-88e0-a63f2b2a27d0.PNG">

### 2. Crop the images:

We **crop** the image to exclude irrelevant data such as the scenery and the car's hood:

`    img = img[60:135,:,:]`

<img width="804" alt="cropcomparison" src="https://user-images.githubusercontent.com/34197007/80828119-0a039500-8be5-11ea-89b4-378534e50a4f.PNG">

### 3. Change color format to YUV:

We will use the **YUV** color format (Y: brightness, UV: chromium to add color) since it's recommended for the nvidia model that we will build (instead of lenet):

`    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)`

<img width="806" alt="yuvcolor" src="https://user-images.githubusercontent.com/34197007/80828105-07a13b00-8be5-11ea-84cd-4d4ac63b412d.PNG">

### 4. Gaussian Blur:

We use **Gaussian Blur** (Gaussian kernel of 3x3) to smoothen the image and reduce noise:

`    img = cv2.GaussianBlur(img,  (3, 3), 0)`

<img width="802" alt="gaussianblur" src="https://user-images.githubusercontent.com/34197007/80828131-0bcd5880-8be5-11ea-8693-ca63a3e6abfb.PNG">


### 5. Resize the images:

We **resize** the image since it helps for faster computations (smaller images are easier to manipulate):

`    img = cv2.resize(img, (200, 66))`

<img width="801" alt="resizecomparison" src="https://user-images.githubusercontent.com/34197007/80828144-0e2fb280-8be5-11ea-8dc1-6f0174b699d6.PNG">


### 6. Normalize the images:

And last, we **normalize** the image by dividing the value of the intensity of the pixel, by the max value (255):

`    img = img/255`

### 7. Apply to all the entire dataset:

To apply the preprocess over all the data we use the **map** function, which iterate through the entire array, and for every element of the array it loops to, it returns an element based on the specified function. It returns it as a list, and we turn it to an array:

```
X_train = np.array(list(map(img_preprocess, X_train)))
X_valid = np.array(list(map(img_preprocess, X_valid)))
```

## VI. Create the CNN model:

First thing to notice is that we are dealing with 1010 training images of 200x66 size.

Also we are dealing with a regression-type problem, instead of classification as with the road signs tutorial. Thus we need **a much more capable model**.

For modeling our steering data based on behavioral cloning, we will create an **Nvidia NN model**, since the LeNet model proved to be insufficient for classifying road images. 

<img width="254" alt="NvidiaCNN" src="https://user-images.githubusercontent.com/34197007/80828139-0cfe8580-8be5-11ea-93b7-afce364f22b5.PNG">

More info regarding the **Nvidia End-to-End learning model** can be found at:
[https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

1. We define a **Sequential** model.

2. We add a **Convolutional** layer of 24 filters of 5x5 kernel size, and a stride of 2x2 (subsample). We don't use padding. We set as input the images we get from preprocessing, and activation function **elu**.
**Notice** that in this tutorial **we use elu instead of relu** activation function because it has the advantage of non-zero gradient in the negative region, which will result in not stacking into a dead relu: A node in the network dies and only feeds its next ones following with zero.

![elu](https://user-images.githubusercontent.com/34197007/80828124-0a9c2b80-8be5-11ea-9064-2379cf21cf78.jpg)


3. We add a second **convolutional** layer of 36 filters of 5x5 kernel size, with stride 2x2 and relu activation function.

4.  We add a third **convolutional** layer of 48 filters of 5x5 kernel size, with stride 2x2 and relu activation function.

5. And the fourth and fifth **convolutional** layers are identical with 64 filters and 3x3 kernel size, with a stride of 1 (since the size decreased sufficiently), and relu activation function.

6. We add a **Dropout** layer to avoid overfitting.

7. We add a **Flatten** layer.

8. We add three more **fully connected (FC)** layers of 100, 50 and 10 nodes respectively, with relu activation function.

9. We add a **Dropout** layer after every FC layer to avoid overfitting.
 
10. We add the last **output** layer for the steering angle.

```
def  nvidia_model():
	model = Sequential()
	model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=(66, 200, 3), activation='relu'))
	model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
	model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(Dropout(0.5))

	model.add(Flatten())
	model.add(Dense(100, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(50, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	optimizer = Adam(lr=1e-3)
	model.compile(loss='mse', optimizer=optimizer)
	return model
```

11. We **compile** the model with **mean squared error** as a loss function, **adam** as an optimizer with 0.001 **learning rate**. We plot the summary below:

<img width="382" alt="summary" src="https://user-images.githubusercontent.com/34197007/80828099-05d77780-8be5-11ea-9240-7ea6dbd8f259.PNG">


## VII. Train the model:

### 1. Train the model:

We train the model with 30 **epochs**, **batch size** of 100 and **verbose** and **shuffle** set to true. 

`history = model.fit(X_train, y_train, batch_size=100, epochs=30, validation_data=(X_val, y_val), shuffle = 1, verbose = 1)`

### 2. Plot the loss graph:

We plot the **loss** graph and we can clearly see the difference between using relu and elu activation function:

<img width="579" alt="relueluloss" src="https://user-images.githubusercontent.com/34197007/80833908-e6921780-8bef-11ea-8bc3-6f53b5aadcc5.PNG">

## VIII. Establish the communication between the model with the simulator:

### 1. Download our model:

We first have to save our model:
`model.save('model.h5')`
And then we download our model:

```
from google.colab import files
files.download('model.h5')
```
### 2. Install the required libraries:

We have to create a bi-directional client-server communication. To do so we need to download all the relevant **libraries**. 

Thus we first create a new **environment** to store these in:

`conda create --name socket`


Initialize a python web application, thus we install and import flask. **Flask** is a python micro framework that is used to build web apps:

``conda install -c anaconda flask``

Then we install and import **socket.io** to initialize the server. Web sockets are used to perform real-time communication between a client and a server:

`conda install -c conda-forge python-socketio`

Install and import **eventlet**, for listening:

`conda install -c conda-forge eventlet`

```
from flask import Flask
app = Flask(__name__) #'__main__'
```

### 3. Set server - web app communication:

We initialize the **server**: 

`sio = socketio.Server()`

We combine the socket server with the flask web app with a **middleware**: 

`    app = socketio.Middleware(sio, app)`

### 4. Send server's requests to the web app:

We use **wsgi** to send the server's requests to the web app, while listening in the `4567` ip address:

`    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)`

### 5. Create the Connect function:

With the event name "Connect" we can trigger the **connect function**: 

`@sio.on('connect')`

The other 2 names that can be used by sio are *message* and *disconnect*.

With the connect function we make sure that our server was connected with the autonomous mode simulator, and as soon as it is connected, we command it to go straight. To do so, we emit actions in the `steering_angle` and the `throttle` commands of the `steer` event of the autonomous mode.

### 6. Get the images from the model:

Then we must connect the `steering_angle` with our model. 

To do so we first **load** the model:

`model = load_model('model.h5')`

Then we declare a function for **listening** from the model, with the event trigger :

`@sio.on('telemetry')`

We first start by going straight and then telemetry will get the predictions of the steering angle by the images dataset and send them back to server. The server then communicates with the client, i.e. the simulation, to control the vehicle's steering angle, such that the cars drives on its own.

In order for our code to make sense of the images, we decode the **base 64** image from the simulator with and we mimic it like a normal file, so that to use it later for processing with `BytesIO`

`image = Image.open(BytesIO(base64.b64decode(data['image'])))`

### 7. Preprocess the images:

However, the image needs to be **preprocessed** the same way as we did with our dataset, before feeding it to the model. Thus, we also include and run the `img_preprocess` function:

```
def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img
```

### 8. Predict the required steering angle:

We **predict** the output of the image using our model:
`steering_angle = float(model.predict(image))`
and we set a speed limit for the throttle, where "speed" is the speed data we get from the simulator:
```
speed = float(data['speed'])
speed_limit = 10
throttle = 1.0 - speed/speed_limit
```
Then we feed this output in the control action of the simulation:
`send_control(steering_angle, throttle)`

## IX. Data Augmented Generator:

After we established the connection, we test the accuracy of our model, by running the connection:

`python drive.py`

and opening the simulator in **autonomous mode**.

Running the simulator in both provided tracks, we notice poor performance due to the small dataset that the model was trained on, resulting in **lack of generalization**. For this reason we must enforce the generalization of the model. Thus, we introduce to the model the **data augmentation** technique.

![dataaugmentation](https://user-images.githubusercontent.com/34197007/80828122-0a9c2b80-8be5-11ea-8fc6-4a7abacb8d94.png)

As discussed in the previous tutorial, data augmentation is the process of modifying the images in several ways, resulting in a variety of types of the images the model is trained on, giving us a bigger dataset and thus a more robust model.

Instead of using the predefined data image generator from keras, we will build our own **batch generator** to add flexibility, variety and make it specific to the application of self-driving car.

### 1. Install libraries:

First install the **imgaug** ([https://imgaug.readthedocs.io/en/latest/](https://imgaug.readthedocs.io/en/latest/)) library, that supports a wide range of augmentation techniques.

`from imgaug import augmenters as iaa`

### 2. Zoom:

We implement the **zoom** augmentation from the `Affine` function which preserves transformations of straight lines and planes within the object, by scaling it in the range of 1-1.3 times:

`zoom = iaa.Affine(scale=(1, 1.3))`

and we apply it to our image:

`image = zoom.augment_image(image)`

We **plot** a random original and the zoomed image to spot the differences:

<img width="807" alt="zoom" src="https://user-images.githubusercontent.com/34197007/80828108-07a13b00-8be5-11ea-873a-3659f655ff92.PNG">

### 3. Pan Augmentation:

Image **panning** is the horizontal or vertical translation of an image.

We set panning in +-10% in either the x or y axis:
```
pan = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
image = pan.augment_image(image)
```

And we again **plot** the original and the panned image:

<img width="800" alt="pan" src="https://user-images.githubusercontent.com/34197007/80828140-0d971c00-8be5-11ea-858b-069213b1bd72.PNG">

### 4. Brightness:

This function is going to **alter the brightness** of the image, making it either lighter or darker by multiplying the intensities of the pixels with a specific value. It can be found inside the `Multiply` function of the image augmentation library. We set the range of 0.2 to 1.2.

```
brightness = iaa.Multiply((0.2, 1.2))
image = brightness.augment_image(image)
```

And we **plot** the original and the brightness altered image:

<img width="804" alt="brightness" src="https://user-images.githubusercontent.com/34197007/80828113-08d26800-8be5-11ea-9170-759e6e70cc2d.PNG">

### 5. Flipping:

With **flipping** we can ensure additional balancing in the distribution of the right and left steering angles dataset.

We can do that by using the `flip` function of `cv2` library and setting the second argument to one for horizontal flipping:

`image = cv2.flip(image,1)`

and flipping the steering angle as well:

`steering_angle = -steering_angle`

We **plot** the original and the flipped image, as well as its corresponding steering angle:

<img width="799" alt="flip" src="https://user-images.githubusercontent.com/34197007/80828130-0b34c200-8be5-11ea-88d5-bd10a93f279f.PNG">

### 6. Randomly apply the augmentations to images:

In order to apply these augmentations at random results, to improve the generalization of the model, we run the above functions at 50% of our dataset's images:

```
def random_augment(image, steering_angle):
    image = mpimg.imread(image)
    if np.random.rand() < 0.5:
      image = pan(image)
    if np.random.rand() < 0.5:
      image = zoom(image)
    if np.random.rand() < 0.5:
      image = img_random_brightness(image)
    if np.random.rand() < 0.5:
      image, steering_angle = img_random_flip(image, steering_angle)
    
    return image, steering_angle
```

We then **plot** the original and the augmented image after we applied the above functions to have an overview, and we get:

<img width="800" alt="augmentedimages" src="https://user-images.githubusercontent.com/34197007/80828110-08d26800-8be5-11ea-9caf-f967e6177db4.PNG">


### 7. Create a batch generator of augmented images:

An important aspect of the data augmented generator is that it creates small batches of augmented images on-the-fly, saving that way memory resources.

Therefore, we create a `batch_generator` function to create the **batch of the augmented images** on the training dataset. We run this function only on the training dataset by using the boolean `istraining`.

The `batch_generator` function is a co-routine, as opposed to the other functions that are sub-routines and this is the reason we use **yield** instead of **return**. `yield` stores all the initialized values inside the `batch_generator`, thus the local values are not re-initialized every time the function is being called. In other words, it keeps the entire function on-hold.

Inside the `batch_generator` function we also include the `preprocess` function to take advantage of the memory benefits of the generator.

### 8. Call the batch generator:

We **request** the next batch generator images from the training and the validation set:
```
x_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, 1))
x_valid_gen, y_valid_gen = next(batch_generator(X_valid, y_valid, 1, 0))
```
and we **plot** an image from the training dataset and an image from the validation set to spot how augmentation techniques altered the training image, whereas the validation image just had been preprocessed:

<img width="798" alt="trainvalidgen" src="https://user-images.githubusercontent.com/34197007/80828100-06700e00-8be5-11ea-9970-90129f091a4d.PNG">

## X. Train the augmented images generator dataset:
 
Now we ensured that the batch generator works as intented, we can move on to **train** the augmented images, by replacing the `fit()` statement with the `fit_generator()`. We also make sure we feed the batches we created instead of the entire dataset, by using the `batch_generator()`.

We set 300 **steps per epoch** and 10 **epochs**, in which each step contains a **batch** of 100 images, thus a total of 30.000 images per epoch which is significantly higher than the size of our initial dataset. We define the validation set and set **200 steps** for it, and set **verbose** and **shuffle** to true.

```
history = model.fit_generator(batch_generator(X_train, y_train, 100, 1),
                                  steps_per_epoch=300, 
                                  epochs=10,
                                  validation_data=batch_generator(X_valid, y_valid, 100, 0),
                                  validation_steps=200,
                                  verbose=1,
                                  shuffle = 1)
```
## XI. Modifications after the Data augmentetaion:

### 1. Delete Dropout layers:

We recall that since we used data augmentation techniques to prevent overfitting, we no more need the **Dropout** layers, so we comment them out. That way we make sure the gap between the training and validation loss is low.

### 2. Change optimizer's learning rate:

We set the Adam optimizer's **learning rate** even lower, to 0.0001.

### 3.  Change samples per bin threshold:

Last, after evaluating our model on the Autonomous mode, we noticed that the vehicle struggles a bit to keep the center of the road, and this is because we may deleted too many of the 0^o^ angle data images. Thus we increase the **threshold** of the samples per bin from 200 to 400:

<img width="280" alt="increasedthreshold" src="https://user-images.githubusercontent.com/34197007/80828135-0c65ef00-8be5-11ea-881f-245b847a8b0f.PNG">

<img width="531" alt="increasedthresholdtrainvalidhisto" src="https://user-images.githubusercontent.com/34197007/80828137-0cfe8580-8be5-11ea-876b-5b3f0a0afbfd.PNG">


## XII. Run the model:

To run the model:
1. We open the **terminal**
2. We head to the path the model is saved in:
`cd .\Behavioral Cloning`
3. We **activate** the conda environment into which we installed the required libraries for the communication of the model with the simulator:
`activate socket`
4. **Run** the `drive.py` program:
`python drive.py`
5. Once it's connected to the port, open the **Udacity simulator** in **Autonomous Mode**.
6. Select both tracks to verify that the model was well generalized and can perform equally sufficient in the track it was trained on, as well as on completely new tracks it never saw before. The result we get, can be seen below:

![](./docs/Test.gif)



**Congratulations! We completed our first autonomous driving Neural Network model!**

