# Copyright 2019 Cloudera, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# # Other models with caret

# To see the list of all models that can be trained
# using the caret package, call the `modelLookup()`
# function, or see the 
# [list of supported models](http://topepo.github.io/caret/train-models-by-tag.html)
# on the caret website.

# You can filter the data frame returned by 
# `modelLookup()` to show only the models that can be
# used for regression tasks.
modelLookup() %>% filter(forReg)


# ## Random forest model with scikit-learn

# Let's see if we can get a better R-squared value by
# using a different model, and also by using a second
# predictor variable in the model.


# ### 0. Preliminaries

# Load the required packages
library(readr)
library(dplyr)
library(caret)


# ### 1. Load data

# Load data representing one brand of chess set ("set A")
chess <- read_csv("data/chess/one_chess_set.csv")


# ### 2. Prepare data

# Split the data into an 80% training set and a 20%
# evaluation (test) set
train_frac <- 0.8
indices <- sample.int(
  n = nrow(chess),
  size = floor(train_frac * nrow(chess))
)
chess_train <- chess[indices, ]
chess_test  <- chess[-indices, ]


# ### 3 and 4. Specify and train model

# This time, specify `method = "rf"` to train a 
# random forest model. Internally, the `train()`
# function calls the `randomForest()` function in the
# randomForest package, so that package must be 
# installed, but you do not need to load it with a 
# `library()` command.

# The formula for a model with two predictor variables
# (`x1` and `y1`) and one response variable (`y`) is
# `y ~ x1 + x2`
model <- train(
  weight ~ base_diameter + height,
  data = chess_train,
  method = "rf"
)


# ### 5. Evaluate model

# Generate predictions from the trained model
test_pred <- predict(model, newdata = chess_test)

# Compute the R-squared
R2(test_pred, chess_test$weight)


# ### Hyperparameter tuning

# To look for an indication of overfitting, see whether
# the R-squared on the training set is higher than the
# R-squared on the test set:
train_pred <- predict(model, newdata = chess_train)
R2(train_pred, chess_train$weight)

# The model does not seem to be overfitting. If it were,
# you could adjust the hyperparameters. For example, to 
# reduce the maximum complexity of the model, you could
# set low values for the `ntree` and `maxnodes` 
# arguments:
#```r
# model3 <- train(
#   weight ~ base_diameter + height,
#   data = chess_train,
#   method = "rf",
#   ntree = 100,
#   maxnodes = 5
# )
#```

# The caret package passes these arguments to the 
# underlying `randomForest()` function.

# To see the full list of hyperparameters that you can
# use when the underlying model function is 
#`randomForest()`, see the R help page for that function:
?randomForest
