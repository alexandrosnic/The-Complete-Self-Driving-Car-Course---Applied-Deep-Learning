# README
This is a tutorial on how to manually create a Neural Network based on the **"The Complete Self Driving Car Course-Applied Deep Learning" course from Udemy.**

**Perceptron:**

The most basic form of Neural Network is Perceptron, a feedforward method.
It receives input and transfer the appropriate output. 

<img width="591" alt="perceptron" src="https://user-images.githubusercontent.com/34197007/79576404-81083c00-80c3-11ea-98dd-beb4f3a222a4.PNG">

Let's see the logics behind it:


## 1. Set the borderline (Classification using a linear model):

For classifying, we get the linear equation:

```
ny=mx+b

=>

mx-ny+b = 0

=>

we set x-> x1 and y-> x2,

m->w1 and n->w2 and b -> bias

=>

w1(x1) + w2(x2) + b(bias value)
```

By using the trained data, to find the w1, w2 and b coefficients of the line that has the minimum error, we can set where the classifier line is. Then, by setting to x1 and x2 our test data, we can classify the point depending if the outcome (score) is positive or negative.

In the concept of neural network, assume that x1, x2 and b=1 is our input, and the linear model is our perceptron (intermidiate node). Then, the x1, x2 and b input will give a specific output based on if it is above or below the line. 



## 2. Classify the data (Activation function):

Our linear model: w1(x1) + w2(x2) + b(bias value) will give a positive or a negative output. However, in order to best classify the dataset, we need an **activation function**. The most common activation function is step function (discrete):

```
y = 1, if score > 0,

y = 0, if score < 0.
```

But this function doesn't tell us how close each data point is in the classifier line (the limit), thus we don't have much information about it.
Thus, we prefer the sigmoid function (continuous) to have a likeliness-probability of each point, instead of binary output.

```
Input: x1, x2, b

First node (linear model): w1(x1) + w2(x2) + b(bias value)

Second node (activation function): y = 1 if score > 0, otherwise 0

Output: Categorization of 1 or 0
```



## 3. Calculate the error (Cross Entropy):

But then we need to find our best linear model. To do this, we use the cross entropy to calculate the **error** of each of the model.

**Cross entropy** is: 

`-Σyln( p ) + (1-y)(ln(1-p))`

Then, the lower the cross entropy, the better linear model.



## 4. Improve the error (Gradient Descent):

To define whether the error is small and use it to improve our weights, we implement **gradient descent**. Gradient descent makes use of back propagation. Back propagation defolds the meaning behind a N.N. 
Thus, the above equation will improve our parameters over time:

`[linear model] - [gradient descent(error)]`

where:

`gradient descent(error) = (points*(probability-label)/NumberOfPoints) * **learningRate**`

We use the learning rate (0.01) to improve the linear model just a small step every time.


## Result:

![perceptrongif](https://user-images.githubusercontent.com/34197007/79576399-7f3e7880-80c3-11ea-88e4-74556b80943f.gif)
