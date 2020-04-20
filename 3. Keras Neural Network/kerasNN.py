import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

n_pts = 500
np.random.seed(0) # With this we make sure we generate the same random numbers every time we run the code
Xa = np.array([np.random.normal(13, 2, n_pts),
               np.random.normal(12, 2, n_pts)]).T
Xb = np.array([np.random.normal(8, 2, n_pts),
               np.random.normal(6, 2, n_pts)]).T

X = np.vstack((Xa, Xb))
y = np.matrix(np.append(np.zeros(n_pts), np.ones(n_pts))).T

plt.scatter(X[:n_pts,0], X[:n_pts,1])
plt.scatter(X[n_pts:,0], X[n_pts:,1])

# We will create the most simple neural network, perceptron, using the keras library.
# 1. We first create a dataset of 1000 points, divided in two classes (binary)
# 2. We make use of the Sequential model of the keras library, which is a linear stack of layers.
# In our case we have the input and the output layer, and we use the add method to add them.
# 3. Our neural network contains dense layers (fully connected neural network), meaning that each node
# of every layer is connected to all the nodes of the preceding layer. We set:
# - Output (units) = 1
# - Input (input_shape) = 2
# - Activation function (activation) = sigmoid
# 4. Then we have to select the optimizer with which we will minimize the error. The most common optimizers
# is vanilla in which we subtract the gradient descent from the parameters, for each point
# (like we did in the manual perceptron). However this is very computational intensive since it runs
# throughout all of the point (which in some cases may be millions), and we also need to be careful choosing
# learning rate. Thus, we need a more effective optimizer. Stochastic gradient descent runs only through
# one simple sample each time. Adam is a Stochastic gradient descent method, based on Adagrad and RMSprop
# which computes adaptive learning rate for each parameter

# 5. To configure the learning process we use the compile method:
# - Optimizer: adam
# - Loss Function (How we calculate the error): Binary cross entropy. If we had more than one classes
# we would use classifier cross entropy.
# - Metrics: Similar to the loss function, but their results are not used to train the model, but just
# to evaluate the performance at each epoch.
#
# 6. To train a model to fit our data we use fit method:
# - x: dataset
# - y: Label
# - Verbose: To display the performance of the model through the iterations. 1 displays, 0 no
# - Batch size: An epoch is may too big to feed it to the computer all at once. Thus, we divide our epoch
# to smaller batches. For a dataset of 1000 points, a batch size of 50 would mean it needs 20 iterations
# to run through the entire epoch.
# - Epochs: An epoch refers to when it iterates through the entire dataset and labels it. Few epochs would
# result in underfitting. Too many would result in overfitting. By trial and error and observing the
# performance of our model, we can identify how many epochs the error needs in order to be minimized
# - Shuffle: After every iteration, we shuffle the dataset, and train a subset of it, so that we don't get
# stuck in a local minimum
#
# 7. By drawing the accuracy and loss plots we were able to identify how many epochs our model needs to
# converge, and in general its performance.

# 8. We then plot the decision boundary from the min, to the max point of the horizontal and vertical axis,
# equally spaced over 50 points in total. We do this using linspace, which has 50 by default. Meshgrid takes
# the vector of the span for the x and y axis, and returns a matrix 50 by 50, with repeating rows for x, and
# columns for y. That way we have each of the y to correspond to every x (to build a 3dim matrix of the points
# and their labels). We then use ravel to reduce the dimension of the matrix from 2 to 1 dimensional array.
# With predict, we train all the points of the grid and returns an array of predictions.
# What is plotted then, it's the graph that represents the probability of each point (intensity of the color)
# to be in either of the classes.

# 9. Last, we use a new test point that has not been labelled, to label it using our neural network classifier

model = Sequential()
model.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))
adam=Adam(lr = 0.1 )
model.compile(adam, loss='binary_crossentropy', metrics=['accuracy'])
h=model.fit(x=X, y=y, verbose=1, batch_size=50,epochs=500, shuffle='true')
plt.plot(h.history['accuracy'])

plt.legend(['accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.plot(h.history['loss'])
plt.legend(['loss'])
plt.title('loss')
plt.xlabel('epoch')
def plot_decision_boundary(X, y, model):
    x_span = np.linspace(min(X[:,0]) - 1, max(X[:,0]) + 1)
    y_span = np.linspace(min(X[:,1]) - 1, max(X[:,1]) + 1)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)
plot_decision_boundary(X, y, model)
plt.scatter(X[:n_pts,0], X[:n_pts,1])
plt.scatter(X[n_pts:,0], X[n_pts:,1])

plot_decision_boundary(X, y, model)
plt.scatter(X[:n_pts,0], X[:n_pts,1])
plt.scatter(X[n_pts:,0], X[n_pts:,1])
x = 7.5
y = 5


point = np.array([[x, y]])
prediction = model.predict(point)
plt.plot([x], [y], marker='o', markersize=10, color="red")
print("prediction is: ",prediction)

plt.show()
