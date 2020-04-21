# README

**This is a tutorial on how to create a Neural Network (NN) using Keras.**

We will create the most simple neural network, perceptron, using the keras library, in a few simple steps, in jupyter lab, to have a visual representation.

The code was part of the **"Complete Self-Driving Car Course - Applied Deep Learning"** course of Udemy.

## 1. Create the dataset:

We first create a dataset of 1000 points, divided in two classes (binary). We do that using the random generator, with a fixed seed

<img width="285" alt="initial dataset" src="https://user-images.githubusercontent.com/34197007/79742695-85229c80-8303-11ea-9214-47eefd5dea3a.PNG">

## 2. Create the NN model:

We make use of the Sequential model of the keras library, which is a linear stack of layers.
In our case we have the input and the output layer, and we use the add method to add them.

## 3. Define the type of the layers' interconnection:

Our neural network contains dense layers (fully connected neural network), meaning that each node of every layer is connected to all the nodes of the preceding layer.

`Sequential().add(Dense(units=1, input_shape=(2,), activation='sigmoid'))`

We set:
- Output (units) = 1
- Input (input_shape) = 2
- Activation function (activation) = sigmoid.

![Dense layer](https://user-images.githubusercontent.com/34197007/79742693-848a0600-8303-11ea-98d9-b765ef365f33.png)

## 4. We set the optimizer of the model:

Then we have to select the optimizer with which we will minimize the error. The most common optimizers is vanilla in which we subtract the gradient descent from the parameters, for each point (like we did in the manual perceptron). However this is very computational intensive since it runs throughout all of the point (which in some cases may be millions), and we also need to be careful choosing learning rate.

 Thus, we need a more effective optimizer. Stochastic gradient descent runs only through one simple sample each time.
  Adam is a Stochastic gradient descent method, based on Adagrad and RMSprop which computes adaptive learning rate for each parameter

`adam=Adam(lr = 0.1 )`

## 5.Compile:

To configure the learning process we use the compile method:
`Sequential().compile(adam, loss='binary_crossentropy', metrics=['accuracy'])`
- **Optimizer:** adam
- **Loss Function (How we calculate the error):** Binary cross entropy. If we had more than one classes we would use classifier cross entropy.
- **Metrics:** Similar to the loss function, but their results are not used to train the model, but just to evaluate the performance at each epoch.

## 6. Fit our data:

To train a model to fit our data we use fit method:

`Sequential().fit(x=X, y=y, verbose=1, batch_size=50,epochs=500, shuffle='true')`
- **x:** dataset
- **y:** Label
- **Verbose:** To display the performance of the model through the iterations. 1 displays, 0 not.
- **Batch size:** An epoch is may too big to feed it to the computer all at once. Thus, we divide our epoch to smaller batches. For a dataset of 1000 points, a batch size of 50 would mean it needs 20 iterations
to run through the entire epoch.
- **Epochs:** An epoch refers to when it iterates through the entire dataset and labels it. Few epochs would result in underfitting. Too many would result in overfitting. By trial and error and observing the
performance of our model, we can identify how many epochs the error needs in order to be minimized.
- **Shuffle:** After every iteration, we shuffle the dataset, and train a subset of it, so that we don't get stuck in a local minimum.

## 7. Accuracy and loss plot:

By drawing the accuracy and loss plots we were able to identify how many epochs our model needs to converge, and in general its performance. By the plots we can see that around 10 epochs are enough for our model to converge.

<img width="292" alt="accuracy" src="https://user-images.githubusercontent.com/34197007/79742700-85bb3300-8303-11ea-8770-30f266b1caf1.PNG">

<img width="293" alt="loss" src="https://user-images.githubusercontent.com/34197007/79742696-85229c80-8303-11ea-99ee-ba50df1b9603.PNG">

## 8. Decision boundary plot:

We then plot the decision boundary from the min, to the max point of the horizontal and vertical axis, equally spaced over 50 points in total. We do this using ***linspace***, which has 50 by default. Meshgrid takes
the vector of the span for the x and y axis, and returns a matrix 50 by 50, with repeating rows for x, and columns for y. That way we have each of the y to correspond to every x (to build a 3dim matrix of the points and their labels).
We then use ravel to reduce the dimension of the matrix from 2 to 1 dimensional array.
With predict, we train all the points of the grid and return an array of predictions.
What is plotted then, it's the graph that represents the gradient of the probability of each point (intensity of the color) to be in either of the classes.

<img width="291" alt="decisionboundary" src="https://user-images.githubusercontent.com/34197007/79742691-83f16f80-8303-11ea-8f47-bad96669b2eb.PNG">

## 9. Predict an unlabelled new data point:

Last, we use a new test point that has not been labelled, to label it using our neural network classifier.

`Sequential().predict(point)`

<img width="283" alt="prediction" src="https://user-images.githubusercontent.com/34197007/79742697-85bb3300-8303-11ea-98ba-3a5536bdbbd1.PNG">
