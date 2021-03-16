from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SelectField
from wtforms.validators import DataRequired

class CarForm(FlaskForm):
    Year = IntegerField('Enter year of first purchase', validators = [DataRequired()])
    Present_Price = StringField('Enter the showroom price', validators = [DataRequired()])
    Kms_Driven = IntegerField('Enter the number of Kilometers driven', validators = [DataRequired()])
    Owner = IntegerField('Enter number of previous owners', validators = [DataRequired()])
    Fuel_Type = SelectField('Choose the fuel type', 
                            choices = [('Petrol', 'Petrol'), ('Diesel', 'Diesel'), ('CNG', 'CNG')], 
                            validators = [DataRequired()])
    Seller_Type = SelectField('Choose if you are an individual or dealer',
                              choices = [('Individual', 'Individual'), ('Dealer', 'Dealer')],
                              validators = [DataRequired()])
    Transmission_Type = SelectField('Choose transmission type',
                              choices = [('Manual', 'Manual'), ('Automatic', 'Automatic')],
                              validators = [DataRequired()])


