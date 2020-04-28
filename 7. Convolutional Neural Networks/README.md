# README

In this tutorial we are going to discuss the main aspects regarding Convolutional Neural Networks (CNN), and show a code demonstration of it. 
I will use Google collab for the visual representation rather than Jupyter lab, since it offers free computational power.

The code was part of the  **"Complete Self-Driving Car Course - Applied Deep Learning" course of Udemy.**

The first part of the tutorial is a brief description of the convolutional neural network whereas the second part is the implementation of the code.

![convolutional](https://user-images.githubusercontent.com/34197007/80314961-0fcc3580-87f5-11ea-9e36-b6726094c059.jpeg)


## Convolutional Neural Networks (CNN)

### Why CNN?
CNNs are perhaps the most powerful Deep Learning architecture since they are used for various applications such as Face Recognition, Object Detection or power self driving cars.

They are known to process data with known grid-like topology, making image recognition computationally manageable.

Two important advantages of CNNs are that they use spatial data for their benefit and **require less parameters** than regular ANNs. 

For example, in an RGB 72x72=5184 image, the RGB pixels would correspond to 15552 different weights for each node of a regular ANN. Since it is a difficult image to learn, we would also need to increase the number of hidden layers as well as the nodes. All these increase the complexity and thus the computational power needed to train the models, making regular ANNs ineffecient. Whereas in CNN, every node is only connected to a small (instead of entire) region of the input volume.

Also, regular NNs often have overfitting problems, whereas the pooling layers of CNN help to avoid overfitting. 

These are the reasons and the motivation behind convolutional neural networks, that can be trained on much larger and colourful images.

### CNN Architecture

CNN takes the input (usually the value of the intensity of pixels), runs through **convolutional**, **pooling**  and **fully connected (FC)** layers, and then pass it through a softmax activation function to classify the input.

**1. Convolutional layer:**

The purpose of the convolutional layer is to **extract and learn specific image features** that can help classify the image. The feature detector is the **kernel matrix** (or convolution filter) of small dimensionality (eg 3 by 3) that shifts through the image in steps known as stride. Stride=1 means the kernel shifts one pixel at a time. The bigger the stride, the smaller the corresponding feature map.

<img width="605" alt="kernel" src="https://user-images.githubusercontent.com/34197007/80314968-12c72600-87f5-11ea-91a2-5c3262a6f686.PNG">

**Feature map** contains specific feature of interest preserved and extracted from the original image. Different kernel matrices or filters are able to detect different features (eg edges, diagonal lines, crosses, curves etc). And the more the kernels we run through the image (larger depth), the more features we learn. Combining all the different feature maps from the different kernels, and passing them through an activation function, gives us the **final output of the convolutional layer: filtered images that are then passed on to the pooling layer.** 

![filters](https://user-images.githubusercontent.com/34197007/80314963-10fd6280-87f5-11ea-8941-0038a3d902fe.png)

An example of the original image, and its corresponding necessary kernel matrices is the above image, that shows that for an "X" image, the filters of up-diagonal, down-diagonal and X are needed:

<img width="892" alt="ximage" src="https://user-images.githubusercontent.com/34197007/80314980-15c21680-87f5-11ea-950e-41b87f6acc9d.PNG">

For 2d images (grayscale) we only have kernels of depth 1. However for 3d images (RGB, 3 channels), the kernel also must be a depth of 3.

<img width="646" alt="3dkernel" src="https://user-images.githubusercontent.com/34197007/80314983-15c21680-87f5-11ea-9f59-f818b9ebd5e2.PNG">

The value of the kernel's weights are learnt during the training process through gradient descent algorithms to minimize the error, in order to learn features of the image. The area that a kernel matrix takes on the original image, is known as **receptor field**. 

![kernelmultiplication](https://user-images.githubusercontent.com/34197007/80314969-12c72600-87f5-11ea-9f8a-43652f42dbdb.gif)

Overall, the process of a convolutional layer is:
- Every cell of the receptor field is **multiplied** by the corresponding kernel matrix cell. 
- We take the sum of these values and divide it with the number of matrix's cells to take the **average**. 
- The result is being shown in the feature map.
- The combination of the feature maps are passed from a **Relu activation function**, accounting for non-linearity, which empirically showed to perform better than sigmoid or tanh, inspired from biological processes. The reason is that sigmoid (and tanh) activation function has the vanishing gradient problem, since its derivative will only be between 0 - 0.25, resulting in very slow convergence of the weights during the gradient descent method. Whereas the derivative of ReLu function is 0-1. The different activation functions are:

![relu](https://user-images.githubusercontent.com/34197007/80314977-15298000-87f5-11ea-9ef7-ecffab8bf42a.png)


**2. Pooling layer:**

The pooling layer **shrinks the filtered images** stack by reducing the dimensionality of the representation of each feature map, reducing the number of parameters and thus the computational complexity of the model, but preserving the most important information.

Pooling **helps to avoid overfitting** using a **pooling operation** (sum, average, max). For example the max pooling uses a kernel of eg 2 by 2 dimensions, and convolves it through the feature map. In every stride (of 2), it keeps the maximum value of the kernel, thus of its corresponding local neighborhood. That way it scales (down) into an abstracted form of the original feature map, preserving the general pattern.

![pooling](https://user-images.githubusercontent.com/34197007/80314974-1490e980-87f5-11ea-8e98-a8f0d14861f3.png)

The deeper the convolutional layer is, the more complex and sophisticated the feature it learns, and the more unrecognizable (for us) the image will be. However it also contains more important information about that specific feature it investigates (features are preserved). Thus, the first block of convolutional layer may learn edges, corners etc, whereas the last block may learn noses and ears. The same apply for the filters: the deeper, the more complex and sophisticated the filters (kernels) are, since they combine previous information at each step.

A nice tool to play with a CNN is: [https://www.cs.ryerson.ca/~aharley/vis/conv/flat.html](https://www.cs.ryerson.ca/~aharley/vis/conv/flat.html)

**3. Fully Connected layers:**

The first part of convolutional and pooling layers is responsible for features extraction, whereas the second part of **the fully connected (FC) layers is responsible for the classification**.

The FC layers work the same way as the multi-layer perceptron-based NN. Thus its difference is that each node is connected with all the nodes of the previous layers. It takes as input the feature matrices of the convolutional layers, but flatten so that they are 1-dimensional.

**In total:**
- **Random values** are being assigned as parameters (weights) for all the nodes of the convolutional, pooling and FC layers.
- Then, the input image **passes through the CNN**, scaled down for feature extraction and based on them the image is classified.
- Based on if the classification is correct, the cross entropy  is calculated and in order to **minimize the error**, the parameters (weights and biases in FC, values of the filter matrix in convolutional layers) are recalculated using **backpropagation** (and gradient descent). However the filters, kernel size, the depth of the convolutional and pooling layers have to be predefined.

All the procedure can be seen in the above image:

<img width="905" alt="cnnexample" src="https://user-images.githubusercontent.com/34197007/80314987-16f34380-87f5-11ea-8e58-1f00dcab22ea.PNG">

## Code Implementation

We will use the MNIST dataset to show an example of how to classify a handwritten image using Convolutional Neural Network. 

***Note:** To make use of google colab free gpu select Runtime > Change Runtime type > GPU.*

### 1. Prepare the dataset:

To prepare the data we first **normalize** them and do **one hot encoding** on the labels. To see more on how we do that, checkout the previous tutorial of MNIST NN.

The difference however is that this time we don't flatten the images into one array, since we feed them as whole images to the CNN. We also add a third dimension, depth of one to illustrate that we are handling one channel image (grayscale):

`X_train = X_train.reshape(60000, 28, 28, 1)`
`X_test = X_test.reshape(10000, 28, 28, 1)`

### 2. Define the LeNet model:

There are various pre-built **CNN architectures** such as LeNet, AlexNet, ZFNet, GoogleNet. We will make use of **LeNet** model to classify our data, which we have already analyzed.

LeNet Model:

![lenet](https://user-images.githubusercontent.com/34197007/80314970-135fbc80-87f5-11ea-83e2-cbd0aa4857a3.png)

GoogleNet model:

![googlenet](https://user-images.githubusercontent.com/34197007/80314966-122e8f80-87f5-11ea-85eb-7a7f424a21ef.png)

AlexNet model:

![alexnet](https://user-images.githubusercontent.com/34197007/80314986-16f34380-87f5-11ea-8dd6-8b2598781097.jpeg)

- We first import the ***Flatten*** library to flatten our data, and ***Conv2D*** and ***MaxPooling2D*** library for the convolutional and pooling layers. 

- We create a ***Sequential*** model. 

`  model = Sequential()`

- We ***add a Conv2D layer*** of 30 filters which are enough to classify our data and not very demanding regarding computational power. Each filter is a 5x5 kernel matrix with a ***stride*** of 1 (default). It takes as input images of 28x28x1. And we use the ***Relu activation function***. Thus in total we have 780 adjustable parameters (each kernel matrix has 25 values (5x5) and 1 bias. Considering 30 filters, this adds up to 780). We will not include ***padding*** (default), which increases the size of our initial matrix with cells of 0s to preserve dimensionality. The reason is that MNIST dataset contains centered images, thus we are not interested in the outer edges of the images. This will result in 24x24 filters.

`model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))`

![padding](https://user-images.githubusercontent.com/34197007/80314973-13f85300-87f5-11ea-9d51-03b409e90e5b.png)

- We add a ***MaxPooling2D*** with a ***pool_size*** of 2x2, to define the neighborhood. The 30 24x24 filters will be scaled down to 12x12. Thus images size of (12, 12, 30).

`  model.add(MaxPooling2D(pool_size=(2, 2))`

- We add another pair of convolutional layer of 15 3x3 filters with a Relu activation function and pooling layers of 2x2. Each filter is going to be scaled down to (10, 10, 15) because of conv2D. And each feature map is going to be scaled down to (5, 5, 15) because of pooling. Although we have to deal with more parameters than in the previous pair, since we do the same for each of the 30 previous feature maps. In total 15*30*3*3+15 (biases) = 4065 parameters. Thus it demands even more computational power. 

`  model.add(Conv2D(15, (3, 3), activation='relu'))`
 ` model.add(MaxPooling2D(pool_size=(2, 2)))`
 
 The summary of the model and the number of its corresponding parameters is:
 
 <img width="392" alt="summary" src="https://user-images.githubusercontent.com/34197007/80314978-15298000-87f5-11ea-9d6a-bf523a69eb56.PNG">

- Then we **flatten** the image to be 1-dimensional to be fed to the FC layer (multi-layer perceptron) by taking the previous (5, 5, 15) and making it (375, 1).

`model.add(Flatten())`
    
- For the **FC layer**, we add a dense layer of a number of nodes arbitrary selected (we chose 500, more than that would need more computational power, less would be less accurate), and **Relu** activation function.

`model.add(Dense(500, activation='relu'))`

- For the **output layer** we add a dense layer with the same number of nodes, as the number of classes, and activation function the **softmax**.

### 3. Compile the Neural Network:

We compile it using the **Adam** optimizer with **learning rate** 0.1, **categorical cross entropy** as a loss function and **accuracy** as metric.

`model.compile(Adam(lr = 0.01), loss='categorical_crossentropy', metrics=['accuracy'])`

### 4. Train the Neural Network:

- We fit the model with 10 **epochs**, **validation set** of 0.1, **batch size** of 400, and **verbose** and **shuffle** equal to true.

`history=model.fit(X_train, y_train, epochs=10,  validation_split = 0.1, batch_size = 400, verbose = 1, shuffle = 1)`

### 5. Plot the accuracy and loss graphs:

We notice much higher accuracy and less loss using CNN than the regular Neural Network, but also that the accuracy for the validation and the training set is almost the same. However we notice some degree of overfitting, since we used 500 nodes. Whenever validation error is higher than the training error, would mean beginning of overfitting. We will see later how to avoid this.

<img width="289" alt="lossbefore" src="https://user-images.githubusercontent.com/34197007/80314971-13f85300-87f5-11ea-859f-87d01efeeb61.PNG">

<img width="278" alt="accuracybefore" src="https://user-images.githubusercontent.com/34197007/80314984-165aad00-87f5-11ea-9000-623fee7269e3.PNG">

### 6. Predict new unlabelled data:

Our test accuracy is 98.8% which is highly satisfying.

To test our NN on an arbitrary unlabelled image, we take an image of the digit 2:

<img src="https://user-images.githubusercontent.com/34197007/80314967-122e8f80-87f5-11ea-9d50-97e56f445d1d.png" width="200" height="200" />

We pre-process it to be suitable to be fed in our CNN:

<img width="185" alt="preprocessDigit2" src="https://user-images.githubusercontent.com/34197007/80314976-1490e980-87f5-11ea-8b4f-a7d14e6e8cb9.PNG">

And we predict its class:

`print("predicted digit: "+str(model.predict_classes(img)))`

As a class of 2, which is correct.

### 7. Fix overfitting:

In order to fix overfitting, we add another one layer called **Dropout layer** which sets a fraction rate of input units to zero at each update during training. 

`from keras.layers import Dropout`

That way, every time the network updates parameters to minimize the error (towards gradient descent), it **randomly selects some nodes to be turned off** and no longer communicate information along the network. The various combination of nodes that are selected each time, helps the worse performing nodes to fix their parameters, rather than enforce the better performing nodes, reducing that way generalization error and preventing overfitting. 

![dropout](https://user-images.githubusercontent.com/34197007/80314962-1064cc00-87f5-11ea-8576-f0860327cdab.png)

This layer only occurs in the training process, while during the test process we use all of the nodes of the network to combine all of the nodes' independent learning.

Usually Dropout layers are placed in between layers of high number of parameters because are more likely to overfit. Thus we place it in between the FC layers with a fraction rate of 0.5 which is the recommended.

`model.add(Dropout(0.5))`

### 8. Run the updated CNN again:

With the introduction of the Dropout layer, we can see improvement of the network's accuracy:

<img width="289" alt="lossdropout" src="https://user-images.githubusercontent.com/34197007/80314972-13f85300-87f5-11ea-83b7-21f65e5c62f1.PNG">

<img width="283" alt="accuracydropout" src="https://user-images.githubusercontent.com/34197007/80314985-165aad00-87f5-11ea-9152-1ea9afc5ce1e.PNG">

### 9. Visualize the feature maps of the filters:

In order to better understand in what specific feature each filter focus, we make use of the **Model class API**:

`from keras.models import Model`

The Model API allows us to define a model (similarly to the sequential model) by instantiating layers of pre-trained models so that we can to reuse sections of them. We are goint to use it to **visualize** the outputs of our convolutional layers. We set as input the input of the first convolutional layer, whereas as output, the output of the first and the second convolutional layer respectively.

`layer1 = Model(inputs=model.layers[0].input, outputs=model.layers[0].output)`
`layer2 = Model(inputs=model.layers[0].input, outputs=model.layers[2].output)`

By using:

`plt.figure(figsize=(10, 6))
	for i in  range(30):
	plt.subplot(6, 5, i+1)
	plt.imshow(visual_layer1[0, :, :, i], cmap=plt.get_cmap('jet'))
	plt.axis('off')`

we visualize our specific features filters.
We use the **jet** cmap to highlight the different pixel values in our images. Colour red highlights pixels with high intensity and blue for low. The result is:

Filters of the first convolutional layer:

<img width="393" alt="filtersconv1" src="https://user-images.githubusercontent.com/34197007/80314964-1195f900-87f5-11ea-9c07-d7bf181bb723.PNG">

Filters of the second convolutional layer:

<img width="431" alt="filtersconv2" src="https://user-images.githubusercontent.com/34197007/80314965-1195f900-87f5-11ea-825b-11bb74fcd695.PNG">

We can clearly see now that the deeper we go, the more abstract the filter images look like.
