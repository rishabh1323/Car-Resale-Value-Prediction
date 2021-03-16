import pickle
import datetime
import sklearn
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        Year = int(request.form['Year'])
        Present_Price = request.form['Present_Price']
        Kms_Driven = int(request.form['Kms_Driven'])
        Fuel_Type = request.form['Fuel_Type']
        Seller_Type = request.form['Seller_Type']
        Transmission_Type = request.form['Transmission_Type']

        Num_Years = datetime.datetime.now().year - Year
        
        if Fuel_Type == 'Diesel':
            Fuel_Type_Diesel = 1
            Fuel_Type_Petrol = 0
        elif Fuel_Type == 'Petrol':
            Fuel_Type_Diesel = 0
            Fuel_Type_Petrol = 1
        else:
            Fuel_Type_Diesel = 0
            Fuel_Type_Petrol = 0
        
        if Seller_Type == 'Individual':
            Seller_Type_Individual = 1
        else:
            Seller_Type_Individual = 0
        
        if Transmission_Type == 'Manual':
            Transmission_Manual = 1
        else:
            Transmission_Manual = 0

        prediction = model.predict([[Present_Price, Kms_Driven, Num_Years, Fuel_Type_Diesel,
                                    Fuel_Type_Petrol, Seller_Type_Individual, Transmission_Manual]])[0]
        if prediction < 0:
            prediction_text = 'Sorry, this car can\'t be sold under these conditions'
        else:
            prediction_text = 'Your car can be sold for Rs {} Lakhs'.format(round(prediction, 2))
        
        return render_template('index.html', prediction_text = prediction_text)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run('0.0.0.0', debug = True)