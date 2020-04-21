# README

## Deep Neural Network

**This is a tutorial on how to create a Deep Neural Network (NN) using Keras.**

We will create a simple deep neural network using the keras library, in a few simple steps, in jupyter lab, to have a visual representation.

The code was part of the **"Complete Self-Driving Car Course - Applied Deep Learning" course of Udemy.**



First thing to notice, is that sometimes we need a non-linear model to best classify our data.

To do that, we combine 2 perceptrons (the simplest type of NN) into a third one, along with their corresponding weights and a bias, in order to give us a non-linear output of the probability of a point, to be in one class or another.

<img width="876" alt="perceptronCombination" src="https://user-images.githubusercontent.com/34197007/79851891-3558d980-83c6-11ea-98e2-0f51cc1e7f67.PNG">


Thus, the ***feed forward*** process of a deep neural network, is that:

-  We set as input our testing points ( eg. x1, x2).

- Then the weights of the first layer of NN are just the coefficients of our linear equations.

- And in the second layer of NN (hidden layer), they are just the weights of how much each linear equation should influence the output.

<img width="815" alt="2layerdeep" src="https://user-images.githubusercontent.com/34197007/79851881-325de900-83c6-11ea-99e3-a6fc03849f3e.PNG">

<img width="871" alt="2layerdeepeq" src="https://user-images.githubusercontent.com/34197007/79851883-32f67f80-83c6-11ea-97b5-94ee03adac01.PNG">

In a feed forward NN there aren't any feedback loops.

The more the hidden layers, the deeper the neural network is. Then, combination of linear models result in non-linear models, and combination of non-linear models can classify more complex given dataset.

<img width="894" alt="deepNN" src="https://user-images.githubusercontent.com/34197007/79851889-34c04300-83c6-11ea-89bd-f69cd3383b85.PNG">

However, wrong linear models can lead to wrong classification. Therefore, we use ***backpropagation*** to minimize the error of our predictions, to create more accurate linear models. 

In backpropagation, we use the output of our train set. If the output is wrongly classified, then the error increases. After we train the whole dataset, we then backpropagate this error to recalculate the weights (gradient descent).

The error function we use for training the NN, is like in a perceptron, ***cross entropy***.

Thus:

**1. Feedforward to predict all outputs.**

**2. Determine total error with cross_entropy.**

**3. Backpropagation.**

**4. Repeat at some learning rate.**



Now moving on to the implementation code:

## 1. Create the dataset:

In our implementation code we used the sklearn library.
sklearn among others, provides access to various datasets. We will use the make_circles method with:

- **n_samples:** Number of points.

- **random_state:** To seed our random generator.

- **noise:** To assign the noise of the circles. The lower, the more perfect circles will be.

- **factor:** The ratio between the size of the two circles.

<img width="292" alt="dataset" src="https://user-images.githubusercontent.com/34197007/79851887-3427ac80-83c6-11ea-8b0f-e804ba25e710.PNG">

Then we follow the same procedure as with a simple perceptron.



## 2. Create the NN model:

We use the **Sequential** model.



## 3. Add the layers:

We add 4 dense NN in the hidden layer, and then add another 1 layer of one NN for the output.



## 4. Compile the NN:

We use the **adam** optimizer, the **binary cross entropy** loss function, and we set **accuracy** as a metric.



## 5. Fit our data:

We set **verbose** to 1, **batch size** to 20, **epochs** to 100 and **shuffle** to true. However, we will see that we need much less than 100 epochs for the model to converge.



## 6. Accuracy and loss plots:

As we notice, our model converges after around 60 epochs, which seems to be the appropriate number of epochs.

<img width="304" alt="accuracy" src="https://user-images.githubusercontent.com/34197007/79851884-338f1600-83c6-11ea-9a1c-4d8d95587b52.PNG">

<img width="283" alt="loss" src="https://user-images.githubusercontent.com/34197007/79851890-3558d980-83c6-11ea-90fe-dd375b1cc410.PNG">



## 7. Decision boundary plot:

Using the contour tool, we plot the decision boundary of our neural network. The different intensity of the color determines the probability of a point to be in one class or another.

<img width="287" alt="decisionBoundary" src="https://user-images.githubusercontent.com/34197007/79851888-3427ac80-83c6-11ea-9f14-ebbc0e7f61ef.PNG">



## 8. Predict an unlabelled new data point:

To test our neural network, we use an unlabelled data point to evaluate its classification:

<img width="296" alt="prediction" src="https://user-images.githubusercontent.com/34197007/79851892-35f17000-83c6-11ea-887a-16ae5c6f54ec.PNG">
