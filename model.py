import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the dataset
data = pd.read_csv('finance_management_dataset_large.csv')

# Print the column names to verify
print("Columns in the dataset:", data.columns)

# One-hot encode categorical columns
data_encoded = pd.get_dummies(data, columns=['Employment Status', 'Marital Status', 'Dependents'], drop_first=True)

# Features (X) and targets (y)
X = data_encoded.drop(['Income', 'Expenses', 'Savings', 'Loan Amount'], axis=1)  # Drop target columns, keeping the features
y_income = data_encoded['Income']  # Target for Income prediction
y_expenses = data_encoded['Expenses']  # Target for Expenses prediction
y_savings = data_encoded['Savings']  # Target for Savings prediction
y_loan = data_encoded['Loan Amount']  # Target for Loan prediction

# Train-test split for each prediction task
X_train, X_test, y_train_income, y_test_income = train_test_split(X, y_income, test_size=0.2, random_state=42)
X_train, X_test, y_train_expenses, y_test_expenses = train_test_split(X, y_expenses, test_size=0.2, random_state=42)
X_train, X_test, y_train_savings, y_test_savings = train_test_split(X, y_savings, test_size=0.2, random_state=42)
X_train, X_test, y_train_loan, y_test_loan = train_test_split(X, y_loan, test_size=0.2, random_state=42)

# Train the models for each prediction task
model_income = RandomForestRegressor()
model_income.fit(X_train, y_train_income)

model_expenses = RandomForestRegressor()
model_expenses.fit(X_train, y_train_expenses)

model_savings = RandomForestRegressor()
model_savings.fit(X_train, y_train_savings)

model_loan = RandomForestRegressor()
model_loan.fit(X_train, y_train_loan)

# Save the models to disk
joblib.dump(model_income, 'model_income.pkl')
joblib.dump(model_expenses, 'model_expenses.pkl')
joblib.dump(model_savings, 'model_savings.pkl')
joblib.dump(model_loan, 'model_loan.pkl')

print("Models have been trained and saved successfully.")