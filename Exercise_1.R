# clear memories
rm(list = ls())

# Load libraries

library(tidyverse)
library(keras)
library(grid)

# install.packages("tensorflow")
library(tensorflow)
# install_tensorflow()
# library(reticulate)
# install_miniconda()

my_seed <- 29032021

# Exercise 1 - Fashion MNIST data

## Load data

fashion_mnist <- dataset_fashion_mnist()
x_train <- fashion_mnist$train$x
y_train <- fashion_mnist$train$y
x_test <- fashion_mnist$test$x
y_test <- fashion_mnist$test$y

class_names <- c('T-shirt/top',
                  'Trouser',
                  'Pullover',
                  'Dress',
                  'Coat', 
                  'Sandal',
                  'Shirt',
                  'Sneaker',
                  'Bag',
                  'Ankle boot')

## A, Show some example images from the data

### From the Tension flow documentation

par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  img <- x_train[i, , ]
  img <- t(apply(img, 2, rev)) 
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste(class_names[y_train[i] + 1]))
}

# You can see that the images show fashion items such as clothing or others

## B, Train a fully connected deep network to predict items
### Normalize the data similarly to what we saw with MNIST.
### Experiment with network architectures and settings (number of hidden layers, 
###   number of nodes, activation functions, dropout, etc.)
### Explain what you have tried, what worked and what did not. Present a final model.
### Make sure that you use enough epochs so that the validation error starts flattening out - 
###   provide a plot about the training history (plot(history))

### 1, Normalizing the data

#### Scaling the axis so that they are between 0 and 1 

x_train <- array_reshape(x_train, c(dim(x_train)[1], 784)) 
x_test <- array_reshape(x_test, c(dim(x_test)[1], 784)) 

x_train <- as.matrix(x_train) / 255
x_test <- as.matrix(x_test) / 255

####  One-hot encoding for the outcome

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

### Building the first model

model_1 <- keras_model_sequential()
model_1 %>%
  layer_dense(units = 128, activation = 'relu', input_shape = c(784)) %>% # adding 128 nods
  layer_dropout(rate = 0.3) %>% # 30% of the set will be the same
  layer_dense(units = 10, activation = 'softmax') # 10 nods since 10 results

summary(model_1)
# 1000480 = 784 (input features) * 128 (first layer nodes) + 128 (biases)

compile(
  model_1,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# Adding the data

history_1 <- fit(
  model_1, x_train, y_train,
  epochs = 30, batch_size = 128,
  validation_split = 0.3
)

history_1

plot(history_1)

# Looks like the first one does somewhat well with validation set. 
# Let's try changing some of the paramenters to get them closer together

# Adding an additional layer

model_2 <- keras_model_sequential()
model_2 %>%
  layer_dense(units = 250, activation = 'relu', input_shape = c(784)) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 128, activation = 'relu', input_shape = c(784)) %>% #adding another layer
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model_2)

compile(
  model_2,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# Add the data

history_2a <- fit(
  model_2, 
  x_train, y_train,
  epochs = 30,
  batch_size = 128,
  validation_split = 0.3
)

history_2a

plot(history_2a)

# Now we can see that data evens out relatively quickly
# The train and valudate are also very close together

# Let's try running this with 

history_2b <- fit(
  model_2, 
  x_train, y_train,
  epochs = 30,
  batch_size = 250,
  validation_split = 0.3
)

history_2b

plot(history_2b)

# Looks like adding a larger batch did help increase the Accuracy

# Larger dropout rate

model_3 <- keras_model_sequential()
model_3 %>%
  layer_dense(units = 250, activation = 'relu', input_shape = c(784)) %>%
  layer_dropout(rate = 0.5) %>% # changing it so 50% of the sample is the same
  layer_dense(units = 128, activation = 'relu', input_shape = c(784)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model_3)

compile(
  model_3,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history_3 <- fit(
  model_3, 
  x_train, y_train,
  epochs = 30,
  batch_size = 128,
  validation_split = 0.3
)

history_3

plot(history_3)

# This seems much less overfit

# Lets try changing the activation function

model_4 <- keras_model_sequential()
model_4 %>%
  layer_dense(units = 250, activation = 'softmax', input_shape = c(784)) %>%
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 128, activation = 'softmax', input_shape = c(784)) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model_4)

compile(
  model_4,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history_4 <- fit(
  model_4, 
  x_train, y_train,
  epochs = 30,
  batch_size = 128,
  validation_split = 0.3
)

history_4

plot(history_4)

# Looks like this is actually worse than the other activation functions

model_5 <- keras_model_sequential()
model_5 %>%
  layer_dense(units = 250, activation = 'relu', input_shape = c(784)) %>%
  layer_dropout(rate = 0.01) %>% 
  layer_dense(units = 128, activation = 'relu', input_shape = c(784)) %>%
  layer_dense(units = 250, activation = 'relu', input_shape = c(784)) %>%  
  layer_dense(units = 10, activation = 'softmax')

summary(model_5)

compile(
  model_5,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history_5 <- fit(
  model_5, 
  x_train, y_train,
  epochs = 30,
  batch_size = 128,
  validation_split = 0.3
)

history_5

plot(history_5)

# This one does great on training, but looking at the validation set, it seems overfit

# Loweting the dropout rate made the model worse

summary <- data.frame(
  model1 = mean(history_1$metrics$val_accuracy),
  model2a = mean(history_2a$metrics$val_accuracy),
  model2b = mean(history_2b$metrics$val_accuracy),
  model3 = mean(history_3$metrics$val_accuracy),
  model4 = mean(history_4$metrics$val_accuracy),
  model5 = mean(history_5$metrics$val_accuracy)
)

summary

# Model 4 is definitely overfit. Model 2 with the larger batch size did the best.

# C, Evaluate the model on the test set

evaluate(model_2, x_test, y_test, batch_size = 250)

# It is somewhat lower in the test set but that is expected. Still very very close though

# D, Convulated network

## Reload the data so it is not scaled

x_train <- fashion_mnist$train$x
y_train <- fashion_mnist$train$y
x_test <- fashion_mnist$test$x
y_test <- fashion_mnist$test$y

# Scaling the data

x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))

# rescale
x_train <- x_train / 255
x_test <- x_test / 255
# one-hot encoding of the target variable
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

cnn_model_1 <- keras_model_sequential()
cnn_model_1 %>%
  layer_conv_2d(
    filters = 32,
    kernel_size = c(3, 3),
    activation = 'relu',
    input_shape = c(28, 28, 1)
  ) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 16, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

summary(cnn_model_1)

compile(
  cnn_model_1,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history_6 <- fit(
  cnn_model_1, x_train, y_train,
  epochs = 30, batch_size = 128,
  validation_split = 0.2
)

history_6

plot(history_6)

# This model already shows great results, although it may be a bit overfit

# The numbers look better on the accuracy but the plot still shows that the training
# and validation are pretty far apart.

# Let's again try more layers

cnn_model_2 <- keras_model_sequential()
cnn_model_2 %>%
  layer_conv_2d(
    filters = 32,
    kernel_size = c(3, 3),
    activation = 'relu',
    input_shape = c(28, 28, 1)
  ) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 16, activation = 'relu') %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

summary(cnn_model_2)

compile(
  cnn_model_2,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history_7 <- fit(
  cnn_model_2, x_train, y_train,
  epochs = 30, batch_size = 128,
  validation_split = 0.2
)

history_7

plot(history_7)

# While this validation error is lower, it shows less signs of overfitting

# Looks like the improvement here is milimal

# Lets try a different activation function for one of the layers

cnn_model_3 <- keras_model_sequential()
cnn_model_3 %>%
  layer_conv_2d(
    filters = 32,
    kernel_size = c(3, 3),
    activation = 'relu',
    input_shape = c(28, 28, 1)
  ) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 50, activation = 'softmax') %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 50, activation = 'softmax') %>%
  layer_dense(units = 10, activation = 'softmax')

summary(cnn_model_3)

compile(
  cnn_model_3,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history_8 <- fit(
  cnn_model_3, x_train, y_train,
  epochs = 30, batch_size = 128,
  validation_split = 0.2
)

history_8

plot(history_8)

# That lowered the results in the train but made it much better in the validation.
# However, other models still do better.

# Try running model 2 with a larger batch size

history_9 <- fit(
  cnn_model_2, 
  x_train, y_train,
  epochs = 30, 
  batch_size = 250,
  validation_split = 0.2
)

history_9

plot(history_9)

# This mode is not doing well, it is very overfit to the training data.

cnn_model_4 <- keras_model_sequential()
cnn_model_4 %>%
  layer_conv_2d(
    filters = 32,
    kernel_size = c(3, 3),
    activation = 'relu',
    input_shape = c(28, 28, 1)
  ) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.01) %>%
  layer_flatten() %>%
  layer_dense(units = 100, activation = 'relu') %>%
  layer_dropout(rate = 0.01) %>%
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

summary(cnn_model_4)

compile(
  cnn_model_4,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history_10 <- fit(
  cnn_model_4, x_train, y_train,
  epochs = 30, batch_size = 128,
  validation_split = 0.2
)

history_10

plot(history_10)

# This model is also overfitting with the accuracy barely reaching 90 in the validation set

summary_2 <- data.frame(
  model6 = mean(history_6$metrics$val_accuracy),
  model7 = mean(history_7$metrics$val_accuracy),
  model8 = mean(history_8$metrics$val_accuracy),
  model9 = mean(history_9$metrics$val_accuracy),
  model10 = mean(history_10$metrics$val_accuracy)
)

summary_2

# The best is actually model 10 even with the horrible curve

evaluate(cnn_model_4, x_test, y_test)

# This is a significant improvement over the fully connected datasets
