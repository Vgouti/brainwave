from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import matplotlib.pyplot as plt
import io
import base64
from prophet import Prophet

app = Flask(__name__)

# Load datasets
quantity_data = pd.read_csv('data/quantity_data.csv')
trends_data = pd.read_csv('data/trends_data.csv')

def prepare_data():
    # Merge datasets on the common column (e.g., 'date')
    merged_data = pd.merge(quantity_data, trends_data, on='date')
    X = merged_data[['trend']].values  # Features (e.g., trend values)
    y = merged_data['quantity'].values  # Target variable (e.g., quantities)
    return X, y

def train_model(model_name, X, y):
    if model_name == 'linear':
        model = LinearRegression()
    elif model_name == 'random_forest':
        model = RandomForestRegressor()
    elif model_name == 'svr':
        model = SVR()
    elif model_name == 'xgboost':
        model = xgb.XGBRegressor()
    else:
        raise ValueError("Invalid model name")
    
    model.fit(X, y)
    return model

def future_prediction_prophet():
    # Prepare data for Prophet model
    data = pd.merge(quantity_data, trends_data, on='date')
    data.rename(columns={'date': 'ds', 'quantity': 'y'}, inplace=True)
    
    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(data)
    
    # Create a dataframe for future predictions
    future = model.make_future_dataframe(periods=30)  # Predict for 30 days into the future
    forecast = model.predict(future)
    
    # Plot the forecast
    fig = model.plot(forecast)
    
    # Save plot to a string in base64 format
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return plot_url

@app.route('/api/prediction', methods=['GET'])
def get_prediction():
    model_name = request.args.get('model', default='linear', type=str)
    X, y = prepare_data()
    model = train_model(model_name, X, y)

    # Predict future quantities (for simplicity, predict the same trend values)
    future_trends = np.array([i for i in range(1, 21)]).reshape(-1, 1)
    predicted_quantities = model.predict(future_trends)

    # Create a DataFrame with the predictions
    future_data = pd.DataFrame({
        'trend': future_trends.flatten(),
        'predicted_quantity': predicted_quantities
    })

    return jsonify(future_data.to_dict(orient='records'))

@app.route('/api/plot', methods=['GET'])
def get_plot():
    model_name = request.args.get('model', default='linear', type=str)
    X, y = prepare_data()
    model = train_model(model_name, X, y)
    
    # Predict future quantities (for simplicity, predict the same trend values)
    future_trends = np.array([i for i in range(1, 21)]).reshape(-1, 1)
    predicted_quantities = model.predict(future_trends)

    # Create a plot
    plt.figure()
    plt.plot(future_trends, predicted_quantities, marker='o')
    plt.title(f'Quantity Prediction using {model_name.capitalize()}')
    plt.xlabel('Trend')
    plt.ylabel('Predicted Quantity')

    # Save plot to a string in base64 format
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return jsonify({'plot_url': f'data:image/png;base64,{plot_url}'})

@app.route('/api/prophet_plot', methods=['GET'])
def get_prophet_plot():
    plot_url = future_prediction_prophet()
    return jsonify({'plot_url': f'data:image/png;base64,{plot_url}'})

if __name__ == '__main__':
    app.run(debug=True)
