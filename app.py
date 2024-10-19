from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained models
model_income = joblib.load('model_income.pkl')
model_expenses = joblib.load('model_expenses.pkl')
model_savings = joblib.load('model_savings.pkl')
model_loan = joblib.load('model_loan.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    prediction_option = request.form['prediction_option']

    # Collecting form data
    income = float(request.form['income'])
    expenses = float(request.form['expenses'])
    savings = float(request.form['savings'])
    loan_amount = float(request.form['loan_amount'])
    credit_score = float(request.form['credit_score'])
    investments = float(request.form['investments'])
    age = float(request.form['age'])
    employment_status = request.form['employment_status']
    marital_status = request.form['marital_status']
    dependents = float(request.form['dependents'])

    # One-hot encoding for categorical data (employment_status, marital_status)
    employment_status_encoded = 1 if employment_status.lower() == "employed" else 0
    marital_status_encoded = 1 if marital_status.lower() == "married" else 0

    input_data = np.array([[income, expenses, savings, loan_amount, credit_score,
                            investments, age, employment_status_encoded,
                            marital_status_encoded, dependents]])

    # Predicting based on the selected option
    if prediction_option == 'income':
        prediction = model_income.predict(input_data)
        result = f"Predicted Income: {prediction[0]:.2f}"
    elif prediction_option == 'expenses':
        prediction = model_expenses.predict(input_data)
        result = f"Predicted Expenses: {prediction[0]:.2f}"
    elif prediction_option == 'savings':
        prediction = model_savings.predict(input_data)
        result = f"Predicted Savings: {prediction[0]:.2f}"
    elif prediction_option == 'loan':
        prediction = model_loan.predict(input_data)
        result = f"Predicted Loan Amount: {prediction[0]:.2f}"
    else:
        result = "Invalid prediction option selected."

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
