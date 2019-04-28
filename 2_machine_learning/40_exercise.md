Complete the following exercise to use scikit-learn
or caret to solve a classification problem, to predict
the type of chess pieces based on their base diameter
and height.

## scikit-learn

1. Create a new empty Python code file and copy the
   contents of `30_sklearn_regress.py` into it.

2. In section **0. Preliminaries**, add the following
   additional imports:
   ```python
   from sklearn.preprocessing import LabelEncoder
   from sklearn.tree import DecisionTreeClassifier
   ```

3. In section **2. Prepare data**, replace the code
   that separates the features (x) and targets (y)
   in the training and test datasets with the 
   following:
   ```python
   train_x = train.filter(['base_diameter','height'])
   train_y = train.piece
   test_x = test.filter(['base_diameter','height'])
   test_y = test.piece
   ```

4. Below that, but still in section **2. Prepare data**,
   add the following code to use a `LabelEncoder` to
   encode the character string labels as integers:
   ```python
   encoder = LabelEncoder()
   encoder.fit(chess.piece)
   train_y_encoded = encoder.transform(train_y)
   test_y_encoded = encoder.transform(test_y)
   ```

5. In section **3. Specify model**, replace the existing
   code with the following:
   ```python
   model = DecisionTreeClassifier()
   ```

6. In section **4. Train model**, replace `train_y` with
   `train_y_encoded`.

7. In section **5. Evaluate model**, replace `test_y` with
   `test_y_encoded` and replace `train_y` with 
   `train_y_encoded`.

8. Remove section **6(a). Interpret the model**—this type 
   of classification model cannot be inspected in the
   same way a linear regression model can.

9. In section **6(b). Make predictions**, replace the code
   that assigns the variable `d` with the following:
   ```python
   d = {
     'base_diameter': [27.3, 32.7, 31, 32.1, 35.9, 37.4],
     'height': [45.7, 58.1, 65.2, 46.3, 75.6, 95.4]
   }
   ```

10. At the end of step **6(b). Make predictions**, use the
    `inverse_transform` method of the `LabelEncoder` to
    decode the integer predictions to character string
    labels:
    ```python
    encoder.inverse_transform(predictions)
    ```

## caret

1. Create a new empty R code file and copy the
   contents of `33_caret_regress.R` into it.

2. In section **2. Prepare data**, add the following code
   at the beginning of the section to change the `piece`
   column in the `chess` data frame to a _factor_.
   ```r
   chess <- chess %>% mutate(piece = factor(piece))
   ```

3. In section **3 and 4. Specify and train model**, change
   the model formula to the following:
   ```r
   piece ~ base_diameter + height
   ```

4. In section **5. Evaluate model**, replace the code that
   computes R-squared with this code that displays a
   confusion matrix and other accuracy metrics:
   ```r
   confusionMatrix(test_pred, chess_test$piece)
   ```
