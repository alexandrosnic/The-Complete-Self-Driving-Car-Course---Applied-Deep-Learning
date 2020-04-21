## README

**This is a tutorial on how to create a Deep Neural Network (NN) using Keras.**

We will create a simple deep neural network using the keras library, in a few simple steps, in jupyter lab, to have a visual representation.

The code was part of the **"Complete Self-Driving Car Course - Applied Deep Learning" course of Udemy.**


The difference between a binary deep neural network (NN) and a multiclass NN, lies in 3 parameters:

- **Activation function:**

In the binary classification we used the sigmoid function to classify our dataset, whereas in multi class NN, we make use of the ***Softmax*** activation function:

`P(score m) = exp(m) / Σ exp(n)`

m: is the score of the class we observe

n: is the score of every train set

Recall that since it is a supervised learning method, we have to train our NN on a labelled dataset.

- **Labelling:**

In the binary classification we made use of 0 and 1 to distinguish between the two classes.
In multiclass classification, to ensure independence between the classes we use ***one hot encoding*** (instead of label encoding) to classify each of the dataset to classes. With one hot encoding we have as many rows as the number of our classes and as many columns as the number of our train sets.

![one hot encoding](https://user-images.githubusercontent.com/34197007/79899215-ca7ec100-840c-11ea-8ed0-a97b97534351.jpg)


- **MultiClass Cross entropy:**

Also, in binary classification we used binary cross entropy, whereas now we use multiclass cross entropy:

`Σn Σm ln(pij)`


To implement the code:

## 1. Create the dataset:

We will create the `make_blobs` dataset of the sklearn library.

<img width="286" alt="dataset" src="https://user-images.githubusercontent.com/34197007/79899220-cb175780-840c-11ea-9d08-0e707175549e.PNG">

## 2.  Label the outcome:

In order to transform the output from label encoding (which labels 0, 1 and 2 if there are three classes), to one hot encoding, we use the `to_categorical` method of keras library.

## 3. Create the NN model:

We use the ***sequential*** model, with ***dense*** layers, ***adam*** optimizer, ***categorical cross entropy*** as a loss function and we print the ***accuracy*** metric.

`model = Sequential()
model.add(Dense(5, input_shape=(2,), activation='softmax'))
model.compile(Adam(lr=0.1), 'categorical_crossentropy', metrics=['accuracy'])`

## 4. Fit our data:

We set **verbose** to 1, **batch size** to 50, and **epochs** to 100. 

## 5. Decision boundary plot:

We use `predict_classes` instead of `predict`, for multiclass datasets and by using `contourf`  it will display different classes with a different color.

<img width="282" alt="decisionBoundary" src="https://user-images.githubusercontent.com/34197007/79899212-c9e62a80-840c-11ea-9bcd-13370309ca86.PNG">

## 6. Predict an unlabelled new data point:

To test our neural network, we use an unlabelled data point to evaluate its classification:

<img width="298" alt="prediction" src="https://user-images.githubusercontent.com/34197007/79899218-cb175780-840c-11ea-841c-8cf42b529a14.PNG">
