"""
    This script runs a small Flask app that displays a simple web form for users to insert a sentence they 
    want to classify with the model.

    Inspired by: https://medium.com/shapeai/deploying-flask-application-with-ml-models-on-aws-ec2-instance-3b9a1cec5e13
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd


# We need to initialise the Flask object to run the flask app 
# By assigning parameters as static folder name,templates folder name
app = Flask(__name__, static_folder='static', template_folder='templates')
# We need to load the pickled model file AND the vectorizer to transform the text 
# to make a prediction on an unseen data point - note that the script assumes the pickled files are in
# the samee folder
label_list = pickle.load(open('label_encoder_list.pkl','rb+'))
scaler = pickle.load(open('scaler.pkl','rb+'))
model = pickle.load(open('model.pkl','rb+'))
df = pd.read_csv('train.csv')
name_list = model.feature_names
df = df[name_list]
categorical_columns = df.select_dtypes(include='object')
# generate example df
example_applicant_df = df.iloc[0]
 
# Function to determine feature type
def get_feature_type(column):
    if pd.api.types.is_numeric_dtype(column):
        return 'numerical', None
    else:
        return 'categorical', column.unique().tolist()

# Generate feature_options from DataFrame
feature_options = []

for column in df.columns:
    feature_type, options = get_feature_type(df[column])
    feature_info = {'name': column, 'type': feature_type}
    if feature_type == 'categorical':
        feature_info['options'] = options
    feature_options.append(feature_info)

feature_explanations = {
    "Loan Amount": "loan amount applied",
    "Funded Amount": "loan amount funded",
    "Funded Amount Investor": "loan amount approved by the investors",
    "Term": "term of loan (in months)",
    "Batch Enrolled": "batch numbers to representatives",
    "Interest Rate": "interest rate (%) on loan",
    "Grade": "grade by the bank",
    "Sub Grade": "sub-grade by the bank",
    "Home Ownership": "Type of home ownership",
    "Employment Duration": "Employment duration in hours",
    "Verification status": "whether the information provided by applicant are verified",
    "Loan Title": "Purpose for loan",
    "Debt to Income": "ratio of representative's total monthly debt repayment divided by self reported monthly income excluding mortgage",
    "Delinquency - two years": "number of 30+ days delinquency in past 2 years",
    "Inquires - six months": "total number of inquiries in last 6 months",
    "Open Account": "number of open credit line in representative's - credit line",
    "Public Record": "number of derogatory public records",
    "Revolving Balance": "total credit revolving balance",
    "Revolving Utilities": "amount of credit a representative is using - relative to revolving_balance",
    "Total Accounts": "total number of credit lines available in - representatives credit line",
    "Initial List Status": "unique listing status of the loan - - W(Waiting), F(Forwarded)",
    "Total Received Interest": "total interest received till date",
    "Total Received Late Fee": "total late fee received till date",
    "Recoveries": "post charge off gross recovery",
    "Collection Recovery Fee": "post charge off collection fee",
    "Collection 12 months Medical": "total collections in last 12 months - excluding medical collections",
    "Application Type": "indicates when the representative is an individual or joint",
    "Last week Pay": "indicates how long (in weeks) a representative has paid EMI after batch enrolled",
    "Accounts Delinquent": "number of accounts on which the representative is delinquent",
    "Total Collection Amount": "total collection amount ever owed",
    "Total Current Balance": "total current balance from all accounts",
    "Total Revolving Credit Limit": "total revolving credit limit"
}
feature_explanations = {key: value for key, value in feature_explanations.items() if key in df.columns.tolist()}

@app.route('/', methods=['POST', 'GET'])
def main():
    if request.method == 'GET':
        example_applicant = example_applicant_df.to_dict()
        return render_template('index.html', feature_options=feature_options, example_applicant=example_applicant, feature_explanations=feature_explanations)
    
    if request.method == 'POST':
        # Collecting user input
        features = []
        for feature_info in feature_options:
            feature_name = feature_info['name']
            feature_type = feature_info['type']
            if feature_type == 'categorical':
                categorical_input = request.form.get(feature_name, '')
                features.append(categorical_input)
                print('categorical:', categorical_input)
            elif feature_type == 'numerical':
                numerical_input = request.form.get(feature_name, '')
                print('numerical:', feature_info)
                print(numerical_input, "type:", type(numerical_input))
                features.append(float(numerical_input))
        print(features)
        # Preprocessing user input
        final_features = pd.DataFrame([features], columns=df.columns.tolist())
        
        # label encode the categorical values and standardize all data
        # Convert categorical columns to numeric labels
        i = 0
        for col in categorical_columns:
            print("this is column:", col)
            final_features[col] = final_features[col].replace(label_list[i])
            i += 1
            
        print(final_features)
        final_features = scaler.transform(final_features)
        
        # Making a prediction (using a mock model)
        prediction = model.predict(final_features)
        predicted_label = prediction[0].item()

        print(type(prediction[0]))
        
        # Returning the response to AJAX
        return jsonify({'predicted_label': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)