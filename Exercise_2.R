# clear memories
rm(list = ls())

# Load libraries

library(tidyverse)
library(keras)
library(here)
library(grid)
library(magick)

# tensorflow::install_tensorflow(extra_packages='pillow')

# Load the data
example_image_path <- file.path(here(), "/data/hot-dog-not-hot-dog/train/hot_dog/1000288.jpg")
image_read(example_image_path)

# A, Preprocess the data

img <- image_load(example_image_path, target_size = c(150, 150))
x <- image_to_array(img) / 255
grid::grid.raster(x)

train_datagen <- image_data_generator(rescale = 1/255) 
test_datagen <- image_data_generator(rescale = 1/255) 

image_size <- c(150, 150)
batch_size <- 50

train_generator <- flow_images_from_directory(
  file.path(here(), "data/hot-dog-not-hot-dog/train/"), # Target directory  
  train_datagen,              # Data generator
  target_size = image_size,  # Resizes all images to 150 × 150
  batch_size = batch_size,
  class_mode = "binary"       # binary_crossentropy loss for binary labels
)

test_generator <- flow_images_from_directory(
  file.path(here(), "data/hot-dog-not-hot-dog/test/"),
  test_datagen,
  target_size = image_size,
  batch_size = batch_size,
  class_mode = "binary"
)

# B, Create the convolutional network

hot_dog_model <- keras_model_sequential() 
hot_dog_model %>% 
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3), 
                activation = 'relu',
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 16,
                kernel_size = c(3, 3), 
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 16,
                kernel_size = c(3, 3), 
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 8, activation = 'relu') %>% 
  layer_dense(units = 1, activation = "sigmoid")   # for binary

hot_dog_model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 2e-5),
  metrics = c("accuracy")
)

history <- hot_dog_model %>% fit_generator(
  train_generator,
  steps_per_epoch = 498 / batch_size,
  epochs = 30
)

hot_dog_model_baseline_eval <- as.data.frame(evaluate(hot_dog_model, test_generator))

# C, Add augmentation

xx <- flow_images_from_data(
  array_reshape(x * 255, c(1, dim(x))),
  generator = train_datagen
)

augmented_versions <- lapply(1:10, function(ix) generator_next(xx) %>%  {.[1, , , ]})
# see examples by running in console:
grid::grid.raster(augmented_versions[[1]])

train_datagen_2 = image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

hot_dog_model_2 <- keras_model_sequential() 
hot_dog_model_2 %>% 
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3), 
                activation = 'relu',
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 16,
                kernel_size = c(3, 3), 
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 16,
                kernel_size = c(3, 3), 
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 8, activation = 'relu') %>% 
  layer_dense(units = 1, activation = "sigmoid")   # for binary

hot_dog_model_2 %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 2e-5),
  metrics = c("accuracy")
)

train_generator_2 <- flow_images_from_directory(
  file.path(here(), "data/hot-dog-not-hot-dog/train/"), # Target directory  
  train_datagen_2,              # Data generator
  target_size = image_size,  # Resizes all images to 150 × 150
  batch_size = batch_size,
  class_mode = "binary"       # binary_crossentropy loss for binary labels
)

history_2 <- hot_dog_model_2 %>% fit_generator(
  train_generator_2,
  steps_per_epoch = 498 / batch_size,
  epochs = 30
)

hot_dog_model_2_eval <- as.data.frame(evaluate(hot_dog_model_2, test_generator))


# D, Using a pre-training model

model_imagenet <- application_mobilenet(weights = "imagenet")

img <- image_load(example_image_path, target_size = c(224, 224))  # 224: to conform with pre-trained network's inputs
x <- image_to_array(img)

# ensure we have a 4d tensor with single element in the batch dimension,
# the preprocess the input for prediction using mobilenet
x <- array_reshape(x, c(1, dim(x)))
x <- mobilenet_preprocess_input(x)

# make predictions then decode and print them
preds <- model_imagenet %>% predict(x)
mobilenet_decode_predictions(preds, top = 3)[[1]]

train_datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

image_size <- c(128, 128)
batch_size <- 100  # for speed up

train_generator <- flow_images_from_directory(
  file.path(here(), "data/hot-dog-not-hot-dog/train/"), # Target directory  
  train_datagen,              # Data generator
  target_size = image_size,  # Resizes all images 
  batch_size = batch_size,
  class_mode = "binary"       # binary_crossentropy loss for binary labels
)

test_generator <- flow_images_from_directory(
  file.path(here(), "data/hot-dog-not-hot-dog/test/"), # Target directory  
  test_datagen,
  target_size = image_size,
  batch_size = batch_size,
  class_mode = "binary"
)

base_model <- application_mobilenet(weights = 'imagenet', include_top = FALSE,
                                    input_shape = c(image_size, 3))


predictions <- base_model$output %>% 
  layer_global_average_pooling_2d() %>% 
  layer_dense(units = 16, activation = 'relu') %>% 
  layer_dense(units = 1, activation = 'sigmoid')

pre_train_model <- keras_model(inputs = base_model$input, outputs = predictions)

pre_train_model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 2e-5),
  metrics = c("accuracy")
)

pre_train_model %>% fit_generator(
  train_generator,
  steps_per_epoch = 498 / batch_size,
  epochs = 1,  # takes long time to train more
)

hot_dog_model_baseline_eval <- as.data.frame(evaluate(pre_train_model, test_generator))















