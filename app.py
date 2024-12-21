# importing the libraries
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

#Create an app object using the Flask class. 
app = Flask(__name__)

#Load the trained model. (Pickle file)
with open('models/model.pkl', 'rb') as model_file:
    model, scaler, encoder, numerical_features, categorical_features = pickle.load(model_file)
# the pickle file contains the model, scaler, encoder, numerical and categorical column names

# loading the default webpage with the elements
@app.route('/')
def home():
    return render_template('index.html')

# called when Predict button is clicked
@app.route('/predict',methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    #getting the data from the html file
    input_data = {
        "Company": [request.form['Company']],
        "Product": [request.form['Product']],
        "TypeName": [request.form['TypeName']],
        "Inches": [float(request.form['Inches'])],
        "ScreenResolution": [request.form['ScreenResolution']],
        "Cpu": [request.form['Cpu']],
        "Ram": [int(request.form['Ram'])],
        "Memory": [request.form['Memory']],
        "Gpu": [request.form['Gpu']],
        "OpSys": [request.form['OpSys']],
        "Weight": [float(request.form['Weight'])]}
    
    #Preprocessing done to the data from the frontend
    input_df = pd.DataFrame(input_data)
    encoded_features = encoder.transform(input_df[categorical_features])
    numerical_values = input_df[numerical_features].values
    final_features = np.hstack((numerical_values, encoded_features))
    final_features_scaled = scaler.transform(final_features)
    
    #price is predicted
    prediction = model.predict(final_features_scaled)
    output = round(prediction[0], 2)

    #the new view of the websitee is loaded which contains the predicted price
    return render_template('index.html', prediction_text='Predicted Laptop Price is €{}'.format(output),
                           request_form=form_data)

#launches the app and loads the default website
if __name__ == "__main__":
    app.run()