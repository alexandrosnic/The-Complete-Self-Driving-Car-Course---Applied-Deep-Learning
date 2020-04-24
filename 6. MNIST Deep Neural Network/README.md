# README

In this tutorial I am going to show how to classify image data and in particular handwritten digits, with a deep neural network, using the MNIST dataset. I will use Jupyter lab for a better visual representation

The code was part of the  **"Complete Self-Driving Car Course - Applied Deep Learning" course of Udemy.**

## MNIST

MNIST is a dataset of various handwritten digits, consisting  of 10 classes (numbers 0-9) as an output, however we have 784 input nodes which represent the 28x28 pixels of an image.

## Training, Validation and Test set

Before running through the tutorial, I emphasize the distinction between training, validation and test set.
The training data are divided into training  and validation set (usually 75% - 25%).

- **Training set:** 

Through the training set, we aim to minimize the training error. We use the optimizer to well train and ensure a high-efficiency neural network to have a small training error so that to avoid **underfitting** (gradient descent).

- **Validation set:** 

To avoid **overfitting**, so that the model does not memorizes very specific features (such as one white pixel) of the image, but instead to learn to see the big picture (that it represents the number "3") we must reduce the depth and complexity of our NN, or reduce the number of nodes or the number of epochs. In other words, we must adhust the **hyperparameters** accordingly. 

Hyperparameters, unlike parameters of the model which consist of the bias and the weights, are the parameters that describe the NN such as the learning rate, the number of hidden layers etc, to modify the complexity (over-under-fitting) of the model.
That way, we minimize the test error, and we ensure **generalization** of our NN model over unlabelled data.

Through the validation set, we can tune the hyperparameters of the NN.  

- **Test set:** 

Through the test set, we test the NN and evaluate its performance based on data it never seen before.

**Overall:**
**1.** Use training set to learn the standard parameters (weights, bias).
**2.** Use validation set to fine tune hyperparameters.
**3.** Use test set to evaluate the whole NN model.


## Implementation of the code

### 1. Create the dataset:

`mnist.load_data()` creates a 60.000 images training dataset and 10.000 images test set of 28x28 pixels

### 2. Ensure correct dimensionality of our input:

`assert()` takes a condition as an argument. If the condition results false, then it will show an error, otherwise it will keep running normally. It helps debugging complex codes. In our case, we use it to ensure right dimensionality of the images and their corresponding labels arrays.

### 3. Plot the images data

`tight_layout()` works well along `plt.subplots` so that the plots do not overlap one over another.
We first plot the subplots empty just for a demonstration.

And then we fill them up, each row representing a class, and each column, a different random sample of this class, with these 28x28 values of the pixels using a greyscale colour map `cmap`. The result is:

We did that using a nested for loop.
We deleted the values on the axis by `.axis("off")`
We set the title for each row by `.set_title()`

### 4. Plot a bargraph of the images' classes (statistics):

To keep track of the images in each class, we plot a bargraph with which each bar represents a class.

### 5. Data pre-processing:

- We first implement **one hot encoding**, since it's crucial for multiclass classification making use of `to_categorical` method.

- Then we **normalize** the data by dividing with the number of pixel intensities, 255, in order to have them being between 0 (min value of a pixel) and 1 (max value of a pixel). This scales down our features, to a uniform range and *decreases variance* among the data, which is important for a NN (e^255^, would be much more difficult to handle than e^1^ in the sigmoid and softmax algorithms).

- Finally, for the sake of matrix multiplication in python, instead of multiplying the matrix 28x28 of the images, with their corresponding weights, we **flatten the matrix** to a vector of 28*28=784 values.

### 6. Create the NN model:

As opposed to the previous example where we just had x (x1) and y (x2) as input, now we have 28x28 representing each one of the pixel.

We define a **Sequential** model. We **add** a first layer with 10 nodes using the `.add` method, that take as input the pixels, and activation function the **relu** which empirically performs better in convolutional networks. We do similarly for the next layers. However for the last layer we choose the **softmax** activation function.

### 7. Compile the NN:

We **compile** the NN by `.compile(...)` and we use **Adam** optimizer, **categorical cross entropy** loss function, and **accuracy** for metric.

Using `model.summary()` we can have an overview of the NN model.

**Note:** As we see from the summary, the first layer has 7850 parameters. Thus each node is parametrized by 7850 weights. In an RGB 72x72 pixels image, this would correspond to 15552 different weights for each  node, and that would need a lot of computing power. This is the reason and the motivation behind convolutional neural networks, that can be trained on much larger and colourful images.

### 8. Train the NN:

We train our NN with `.fit(...)`, with 10% as a **validation set** for tuning the hyperparameters by `validation_split`, 10 **epochs**, **batch size** of 200 and **verbose** and **shuffle** set to true.

### 9. Plot training and validation error and accuracy:

We can see that the **validation loss** is consistently lower than the **training loss**, which is rational since the training set is 9 times bigger than the validation set, and validation is being applied to a model that have already been trained.

### 10. Print the test score accuracy and loss:

To perform a final evaluation for images the model haven't seen before, we check the **test loss** score (`score[0]`) using the `.evaluate(...)` and the **test accuracy** score (`score[1]`).
The NN performs very well, but not better than how a convolutional network would.

### 11. Test an unlabelled image:

We get the image of a URL by `requests.get(url)` from the **requests** library, and then to display the raw content we use the **Image** module of the `PIL` (Python Imaging Library) library, with `Image.open(response.raw)` and then with `imshow`. This is the raw image we get:

**Preprocess the image:**

However, we first need to transform the image to have the same as input as the NN requires. Thus:
- To do that, we import the **cv2** library.
- Then we transform it into an **array** by `np.asarray()`.
- We **resize** it to a 28x28 pixels instead of 850x850 as the original image is, by `cv2.resize()`.
- We turn it to a **grayscale** image, thus making it 2-dimensional (28, 28) array, instead of 3-dimensional (28, 28, 4) where 4 corresponds to red, green, blue and alpha channels. We do that by `cv2.cvtColot(image,cv2.COLOR_BGR2GRAY)`.
- We use `cmap` so that it won't see the image as a coloured image.
- **Fix background-foreground:** In order to exchange the colours of white and black, so that it has the same representation as our  we use `cv2.bitwise_not()`.
- **Normalize** the pixels intensity by dividing them with 255.
- And last, **reshape** the 28 by 28 array, to a 1 by 784 array

**Predict the unlabelled image's class:**
We do that by `.predict_classes()`
