# Importing the necessary libraries
import numpy as np  # Used for creating NumPy arrays
import pandas as pd  # Used for working with structured tables (DataFrames)
import matplotlib.pyplot as plt  # Used to make plots and graphs
import seaborn as sns  # Data visualization library for creating plots
from sklearn.model_selection import train_test_split  # Splitting data into train and test sets
from sklearn.linear_model import LinearRegression  # Linear Regression model
from sklearn import metrics  # Used for model evaluation

# Loading the data from a CSV file into a Pandas DataFrame
insurance_dataset = pd.read_csv("G:/ML_Projects/Medical Insurance Cost Prediction/insurance.csv")

# Displaying the first five rows of the DataFrame
insurance_dataset.head()

# Number of rows and columns in the DataFrame
insurance_dataset.shape

# Getting information about the dataset
insurance_dataset.info()

# Checking for missing values in each column
insurance_dataset.isnull().sum()

# Statistical measures of the dataset
insurance_dataset.describe()

# Visualizing the distribution of 'age' values using different plotting methods
sns.set()
plt.figure(figsize=(6,6))
sns.displot(insurance_dataset['age'])
plt.title("Age Distribution")
plt.show()

# Plotting 'age' distribution with another method for comparison
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['age'])
plt.title("Age Distribution")
plt.show()

# Plotting 'sex' distribution using countplot
plt.figure(figsize=(6,6))
sns.countplot(x="sex", data=insurance_dataset)
plt.title("Sex Distribution")
plt.show()

# Encoding categorical features like 'sex', 'smoker', and 'region' into numerical values
insurance_dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
insurance_dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
insurance_dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

# Splitting the Features and Target variables
x = insurance_dataset.drop(columns='charges', axis=1)
y = insurance_dataset['charges']

# Splitting the data into Training and Testing Data
x_train, x_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Model Training using Linear Regression
regressor = LinearRegression()
regressor.fit(x_train, Y_train)

# Model Evaluation on training data
training_data_prediction = regressor.predict(x_train)
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print("R squared value (training data): ", r2_train)

# Model Evaluation on test data
test_data_prediction = regressor.predict(x_test)
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print("R squared value (test data): ", r2_test)

# Building a predictive system for insurance cost
input_data = (60, 1, 36.005, 0, 1, 2)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = regressor.predict(input_data_reshaped)
print("The insurance cost is USD: ", prediction[0])

input_data = (31, 1, 25.74, 0, 1, 0)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = regressor.predict(input_data_reshaped)
print("The insurance cost is USD: ", prediction[0])
