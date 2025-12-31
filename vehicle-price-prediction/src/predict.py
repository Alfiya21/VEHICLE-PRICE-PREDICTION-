import joblib
import pandas as pd

model = joblib.load("model/vehicle_price_model.pkl")

new_vehicle = pd.DataFrame([{
    "make": "Toyota",
    "model": "Camry",
    "engine": "2.5L",
    "cylinders": 4,
    "fuel": "Gasoline",
    "mileage": 30000,
    "transmission": "Automatic",
    "body": "Sedan",
    "doors": 4,
    "drivetrain": "Front-wheel Drive",
    "vehicle_age": 3,
    "mileage_per_year": 10000
}])

predicted_price = model.predict(new_vehicle)
print("Predicted Price:", predicted_price[0])
