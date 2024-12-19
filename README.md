# SVR
Support Vector Regression (SVR) is a machine learning technique used for regression problems, where the goal is to estimate continuous values based on input features. SVR is based on the principles of Support Vector Machines (SVM), where the objective is to find a hyperplane that best fits the data points while minimizing errors within a defined margin.
Problem Overview

Let's assume you have a dataset with input features (X) and target values (Y), and you want to use SVR to predict continuous values for given input features.
Steps to Implement Support Vector Regression (SVR):

    Data Preprocessing: This involves cleaning the data, normalizing features (important for SVR), and splitting it into training and testing sets.
    Model Training: Train an SVR model using the training data.
    Model Evaluation: Evaluate the performance of the model using metrics like Mean Squared Error (MSE) or R².
    Prediction: Predict new values using the trained SVR model.

Below is an example Python code to solve an SVR problem:
Python Code for SVR using scikit-learn

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate some sample data for demonstration
# Let's assume this data is representing some real-world continuous data
# X represents the input features, Y represents the target variable to estimate

# Example: Let's create a simple dataset with a non-linear relationship
X = np.arange(1, 11).reshape(-1, 1)  # Input data (e.g., 1 to 10)
Y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])  # Output data (e.g., squared values)

# Data Preprocessing
# It's important to scale the features for SVR
sc_X = StandardScaler()
sc_Y = StandardScaler()

# Feature scaling (SVR is sensitive to the scale of input features)
X_scaled = sc_X.fit_transform(X)
Y_scaled = sc_Y.fit_transform(Y.reshape(-1, 1)).flatten()  # Flatten Y to make it a 1D array

# Split the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# SVR Model (using radial basis function kernel)
regressor = SVR(kernel='rbf')  # rbf is the default kernel
regressor.fit(X_train, Y_train)

# Predictions
Y_pred = regressor.predict(X_test)

# Reverse the scaling to get the actual predictions
Y_pred_actual = sc_Y.inverse_transform(Y_pred.reshape(-1, 1))
Y_test_actual = sc_Y.inverse_transform(Y_test.reshape(-1, 1))

# Evaluate the model
mse = mean_squared_error(Y_test_actual, Y_pred_actual)
r2 = r2_score(Y_test_actual, Y_pred_actual)

# Print evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Visualize the results
# Plot the SVR results (prediction line) along with the original data points

plt.scatter(X, Y, color='red', label='Actual Data')  # Original data
plt.plot(X, regressor.predict(sc_X.transform(X)), color='blue', label='SVR Model')  # Predicted curve
plt.title('SVR Regression Results')
plt.xlabel('Input Feature')
plt.ylabel('Target Output')
plt.legend()
plt.show()

Explanation of the Code:

    Data Generation:
        X represents the input features (for example, numbers 1 to 10).
        Y represents the target values that follow a non-linear relationship (e.g., squared values).

    Data Scaling:
        SVR is sensitive to the scale of the features. So, we use StandardScaler to standardize both the input features (X) and the target values (Y) before training the model.
        The sc_X and sc_Y are used to scale the features and target, and later the predictions are inverse-transformed to obtain the actual values.

    Model Training:
        An SVR model is created using the SVR class from scikit-learn with the RBF (Radial Basis Function) kernel.
        The model is trained using the scaled training data (X_train and Y_train).

    Prediction and Evaluation:
        After training, the model is used to make predictions on the test data (X_test).
        The Mean Squared Error (MSE) and R² score are calculated to evaluate the model’s performance.
            MSE gives the average squared difference between predicted and actual values. A lower value is better.
            R² Score indicates how well the regression predictions approximate the real data points. A higher value (closer to 1) indicates a better model fit.

    Visualization:
        A scatter plot is used to display the actual data points (X vs Y).
        The predicted values are plotted as a continuous curve over the input features to show the regression output.

Example Output:

For the above code, you might see something like this in the terminal:

Mean Squared Error: 0.0169
R² Score: 0.9987

This indicates that the SVR model has performed very well on the dataset, with a very high R² score close to 1.
Key Points:

    SVR is powerful for non-linear regression problems and is effective when there is a non-linear relationship between the input features and target variables.
    Feature Scaling is crucial when using SVR because it is sensitive to the scale of input features.
    The RBF kernel is often a good choice for capturing complex non-linear relationships between input features and target values.

Customizing the Code:

You can modify the code for your specific problem by:

    Replacing the synthetic dataset (X and Y) with your real dataset.
    Fine-tuning the SVR model by adjusting hyperparameters like C (penalty parameter) and epsilon (margin of tolerance for errors). You can use GridSearchCV to find the best hyperparameters.

This should give you a solid foundation for solving a problem with Support Vector Regression.
