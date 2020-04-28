# README

In this tutorial we are going to classify road signs using Convolutional Neural Networks (CNN).
I will use Google collab for the visual representation rather than Jupyter lab, since it offers free computational power.

The code was part of the  **"Complete Self-Driving Car Course - Applied Deep Learning" course of Udemy.**


## Preprocessing

The road signs are going to be classified into 43 different classes. For this, we will use a dataset which is smaller than the MNIST dataset, contains more classes and larger images of various road signs. The dataset can be found in: 

[https://bitbucket.org/jadslim/german-traffic-signs/src/master/](https://bitbucket.org/jadslim/german-traffic-signs/src/master/)

and we clone it in our code:

`git clone`

First we will need to preprocess the data in order to be suitable to be fed into our LeNet CNN.

The dataset contains **pickle** files which are used to serialize (converts all the objects to character stream which is often more convenient to store and transfer) the data before save them on disk and deserialize (unpickle) them when is needed. 

We open the train, validation and test pickle files for reading in binary format:

`with open('german-traffic-signs/train.p', 'rb') as f:
    train_data = pickle.load(f)`

From these, we only need the `features` and `labels` values.

`X_train, y_train = train_data['features'], train_data['labels']`

The train set contains almost 35000 RGB (3 channels) images of 32x32 size, the validation set 4400 and the test set 12600.

We use `assert` just to check potential errors before starting the process.

We make use of `pandas` library which is a great data analysis tool to manipulate the `signnames.csv` file.

And then we treat it similarly to the MNIST dataset:

`  
  num_of_samples=[]
  cols = 5
  num_classes = 43
  fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5,50))
  fig.tight_layout()
  for i in range(cols):
      for j, row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0,(len(x_selected) - 1)), :, :], cmap=plt.get_cmap('gray'))
        axs[j][i].axis("off")
        if i == 2:
          axs[j][i].set_title(str(j) + " - " + row["SignName"])
          num_of_samples.append(len(x_selected))`

A sample of the dataset's images can be seen below:

We can see that the distribution of the dataset's images is much less uniform than of the MNIST dataset, which may result in less accurate result:


## I. Preprocess the images:

The road signs images are far more difficult challenge to face than the ones of the MNIST dataset since there is a high variety of images with different background, colours, conditions etc. And that's why we are going to preprocess the dataset.

For the better visualization of the preprocess, we will focus on this image:


### 1. Convert to grayscale:

First we will convert the images to a grayscale, since the colors are not very important for the road signs. The edges of the images that really matter are the edges, curves, shape etc. That way we reduce the depth of the images from 3 (RGB) to 1, thus the network requires less parameters, will be more efficient and will require less computing power to classify the data.

`img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`

### 2.  Histogram Equalization:

Histogram Equalization aims to standardize the lighting of our images, by enhancing their contrast, making them all have similar lighting, which can help in feature extraction. 

`img = cv2.equalizeHist(img)`

### 3. Normalization:

We normalize each pixel by dividing with the maximum value of a pixel intensity.

### 4. Apply to all:

To apply these 3 preprocessing procedures, we will make use of `map` function which returns an output for each element of the array, ultimately creating a new array with updated values:

`X_train = np.array(list(map(preprocess, X_train)))`

Then we just display a random image to verify that has gone through the process:

### 5. Add depth dimension:

Before feeding these images to our CNN, we must first add the depth dimension (of 1 since it is grayscale) in order to be in the desired shape of the input of CNNs:

`X_train = X_train.reshape(34799, 32, 32, 1)`

### 6. One hot endocing:

Last, we must apply one hot encoding to the labels of our datasets:
`y_train = to_categorical(y_train, 43)`


## II. Design the Neural Network:

We will use the LeNet Neural Network for classifying the images.

### 1. Define the model:
We define a **Sequential** model.
`model = Sequential()`

### 2. Add the convolutional and pooling layers:

- We add a **convolutional layer** of *60 filters of 5x5 size* (kernel matrix) that have as *input* the preprocessed images of (32, 32, 1) size, and we set *relu* as the activation function. The result are 60 filters of 28x28 size. Since we use 5x5 filters with stride of 1, that's why we lose 2 pixels in each dimension, but this does not affect us since all the important information is around the center of the image. The borders do not contain significant features (otherwise we would use padding). Ultimately, we end up with 60*5*5+60 (biases) = 1560 adjustable parameters.

`model.add(Conv2D(30, (5, 5), input_shape=(32, 32, 1), activation='relu'))`


- We add a **max pooling layer**, of *2x2 size*. The result are 12x12 filters of depth 60.

`model.add(MaxPooling2D(pool_size=(2, 2)))`

- We add **another pair of convolutional and pooling layer**. The second convolutional layer consists of 15 filters of 3x3 size, thus 30*15*3*3 + 15 (biases) = 4065 parameters. The second max pooling layer is again size of 2x2. The result are 6x6 filters of depth 15.

`model.add(Conv2D(15, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))`

### 3. Add a flatten layer:

We add a **Flatten layer** to convert our data to 1-dimensional in order to be fed to the *Fully Connected (FC)* layer, resulting in a 6*6*15=540 nodes array.

`model.add(Flatten())`

### 4. Add the Fully Connected layers:

- We add a **Dense layer** of 500 nodes arbitrary selected (less nodes results in less accuracy, more nodes require more computational power) and **relu** as an activation function.

`model.add(Dense(500, activation='relu'))`

- We add a **Dropout layer** with fraction rate of 0.5 (recommended rate). Recall to the previous tutorial for the usability of this layer.

`model.add(Dropout(0.5))`

- We add the **output layer**. It will be a dense layer of number of nodes equal to the number of classes and **softmax** as the activation function.

`model.add(Dense(43, activation='softmax'))`


### 5. Compile the model:

We **compile** the model with Adam optimizer, categorical cross entropy as loss function and accuracy for metric.

`model.compile(Adam(lr = 0.01), loss='categorical_crossentropy', metrics=['accuracy'])`

Running the code gives us this summary:

### 6. Train the model:

We train the model with **10 epochs**,  **batch size of 400**, we set the validation set and set **verbose and shuffle to true**.

`history = model.fit(X_train, y_train, batch_size=400, epochs=10, validation_data=(X_val, y_val), shuffle = 1, verbose = 1)`

### 7. Analyze how the network performs:

- We plot the accuracy and loss function:

We notice high loss for the training and validation set and relatively not that much high accuracies, implying that the model is not working effectively and that it overfitted our data (validation accuracy lower than training accuracy)

- We run the model on our test data to evaluate its performance and we get an accuracy score of around 91% which is not very good.

This perhaps is due to the non-uniform dataset, comparing to MNIST.

Thus, we must fine-tune our model to improve its performance.

## III. Fine-tune the Neural Network:

We have to tackle the low accuracy and overfitting issues.

Each dataset must be treated differently thus it is always good to make use of trial and error to modify the parameters. Some good modifications are:

### 1. Adam's initiallearning rate adjustment:

Adam optimizer uses individual adaptive learning rates, however we must define a good initial learning rate (lr). High lr may lead to low accuracy, whereas low lr helps the network learn more effectively for complex models.

For initial learning rate 0.001 and by plotting the losses and accuracies again, we notice a bit better performance in training accuracy, but not significant, and still overfitting occurs.

### 2. Increase the number of filters of the convolutional layers:

We double the number of filters in the convolutional layers (from 30 to 60 and from 15 to 30), resulting in this summary:

After plotting loss and accuracy graphs, we notice better performance in the training, validation and test accuracy, however still overfitting occurs.

### 3. Add extra convolutional layers:

This will help extract features more effectively and improve accuracy. We add two extra convolutional layers after the previous ones, with the same dimensions (60 and 30). That way, even though each convolutional layer introduces new extra parameters, the total number of parameters decreased since the FC layers had less input parameters as we see in the summary:

The accuracy increased even more, as well as the overfitting.

### 4. Add an extra Dropout layer:

We add an extra dropout layer after the convolutional and pooling layers, since it helps reduce overfitting.

We solved overfitting issue (validation loss less than training loss - validation accuracy higher than training accuracy), the test accuracy significantly increased, even though the training accuracy decreased.

Thus, the model seems accurate enough to classify non-labelled new data.

### 5. Data Augmentation:

This technique generates new data to be used during our model's training process. It is done by modifying the images of the dataset to look like new images, like rotating, flipping, de-colorizing, changing background etc.

We add these transformations in the preprocessing section.

`from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10.)`

The data generator is not saved in the memory but instead it only generates these augmented images when requested (during the training process and that's why is slower), making it computationally more efficient and less memory intensive.

We add transformations of the width, the height, zoom, shear and rotation. 

- Width and height range values lower than 1 represent persentage whereas bigger than 1 represent number of pixels.

- Zoom range represents a range between [1-range , 1+range].

- Shear range represents angle as well as rotation.

`datagen.fit(X_train)` will create some useful statistics for these images.

We create new data of batch size 15, by:

`batches = datagen.flow(X_train, y_train, batch_size = 15)`

We plot some of these images to see the difference from our initial dataset:

Then we modify the training of our model to train our augmented data instead of the initial data.

`history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=50),
                            steps_per_epoch=2000,
                            epochs=10,
                            validation_data=(X_val, y_val), shuffle = 1)`

- steps per epoch accounts for the amount of batches the generator generates per epoch, i.e. the total data per epoch (size of augmented dataset). Thus with a batch size of 50, and 2000 steps, we have 50*2000=100,000 size dataset.

Last, we delete the extra Dropout layer to decrease the gap between validation and training accuracy, since the generator took action for the overfitting.

Running all the cells again, we notice a very good accuracy of our final model.

## IV. Test the model to new unlabelled data:

After modifying our model to improve the accuracy and avoid overfitting, we will feed it with new unlabelled images from the internet to validate its efficiency. Some sources of random images can be found here:
-   [https://c8.alamy.com/comp/G667W0/road-sign-speed-limit-30-kmh-zone-passau-bavaria-germany-G667W0.jpg](https://c8.alamy.com/comp/G667W0/road-sign-speed-limit-30-kmh-zone-passau-bavaria-germany-G667W0.jpg)
    
-   [https://c8.alamy.com/comp/A0RX23/cars-and-automobiles-must-turn-left-ahead-sign-A0RX23.jpg](https://c8.alamy.com/comp/A0RX23/cars-and-automobiles-must-turn-left-ahead-sign-A0RX23.jpg)
    
-   [https://previews.123rf.com/images/bwylezich/bwylezich1608/bwylezich160800375/64914157-german-road-sign-slippery-road.jpg](https://previews.123rf.com/images/bwylezich/bwylezich1608/bwylezich160800375/64914157-german-road-sign-slippery-road.jpg)
    
-   [https://previews.123rf.com/images/pejo/pejo0907/pejo090700003/5155701-german-traffic-sign-no-205-give-way.jpg](https://previews.123rf.com/images/pejo/pejo0907/pejo090700003/5155701-german-traffic-sign-no-205-give-way.jpg)
    
-   [https://c8.alamy.com/comp/J2MRAJ/german-road-sign-bicycles-crossing-J2MRAJ.jpg](https://c8.alamy.com/comp/J2MRAJ/german-road-sign-bicycles-crossing-J2MRAJ.jpg)

We feed it this image:

We process it to be fed to our model:

And we print its prediction. We notice that after the modifications, our model correctly classifies the image. 



