# Name: Christopher A. Lee
# Date: 02/11/2025
# Prof: Dr. Azarnoosh
# Course: Machine Learning for Data Analysis

# Libraries
import numpy as np
import pandas as pd
import sklearn
import statsmodels.api as sm

print("Problem 3")
# Given dataset
np.random.seed(123)
Price = np.round(np.random.normal(loc=300, scale=50, size=50), 2)
Size = np.round(np.random.uniform(low=800, high=4000, size=50), 2)
Bedrooms = np.random.choice(range(2, 7), size=50, replace=True)
Age = np.random.choice(range(1, 51), size=50, replace=True)
Distance_to_City_Center = np.round(np.random.uniform(low=1, high=30, size=50), 2)
# Create DataFrame
data2 = pd.DataFrame({
    "Price": Price,
    "Size": Size,
    "Bedrooms": Bedrooms,
    "Age": Age,
    "Distance_to_City_Center": Distance_to_City_Center
})
print(data2.head())
data2 = sm.add_constant(data2)

print("\nPart a")
house_price_model = sm.OLS(data2["Price"], data2[["Size", "Bedrooms", "Age", "Distance_to_City_Center", "const"]]).fit()
print(house_price_model.summary())

print("\nPart b")
print("Based on the p-values in the summary table, the only variable that significantly impacts the house price is Age. This is because Age has a p-value of 0.038, which is less than the significance level of 0.05. All other variables (Size, Bedrooms, Distance) have p-values greater than 0.05, meaning they are not statistically significant.\n")

print("Part c")
print(f"House Sales Prediction\nParameters: Size; 2500ft, Bedrooms; 4, Age; 20 years, Distance to City Center; 10 miles\nCost: ${(house_price_model.params['Size'] * 2500) + (house_price_model.params['Bedrooms'] * 4) + (house_price_model.params['Age'] * 20) + (house_price_model.params['Distance_to_City_Center'] * 10) + house_price_model.params['const']:.4f}")