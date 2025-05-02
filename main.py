import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the relevant columns from the Excel sheet
# Note that data.xlsx is a relative path to this file. Make sure to store the xlsx file
# in the same directory as your python file

filePath = "data.xlsx"
df = pd.read_excel(filePath, sheet_name="Web vs Foot Traffic", usecols="C:E", skiprows=4, nrows=16)

# Select features (Total Web Traffic) and target (Web Purchases)

df_stuff = df[["Total Web Traffic", "Web Purchases"]]

# Reshape data for sklearn- dw abt this too much (its mostly linear algebra)

x = np.array(df_stuff['Total Web Traffic']).reshape(-1, 1)
y = np.array(df_stuff['Web Purchases']).reshape(-1, 1)

# Split the data into training (70%) and testing (30%) sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Initialize and train the linear regression model

regr = LinearRegression()
regr.fit(x_train, y_train)

# Calculate and print the R-squared value

r_squared = regr.score(x_test, y_test)
print("The R Squared value is ", str(round(r_squared, 4)))

# Predict using the trained model

y_pred = regr.predict(x_test)

# Prepare a smooth range of x values for plotting the regression line (again, don't worry abt the reshape stuff)

x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
y_pred_range = regr.predict(x_range)

# Get the slope (m) and intercept (b) for the equation

m = regr.coef_[0][0]
b = regr.intercept_[0]

# Plot the training and testing data

plt.scatter(x_train, y_train, color="red", label="Training data")
plt.scatter(x_test, y_test, color="blue", label="Testing data")

# Plot the regression line

equation="y = " , str(round(m, 2)) , "x" + str(round(b, 2))
plt.plot(x_range, y_pred_range, color="black", label=equation)

# Making the graph look pretty + displaying it

plt.xlabel("Total Web Traffic")
plt.ylabel("Web Purchases")

plt.title("Web Traffic vs Purchases")
plt.legend()

plt.show()
