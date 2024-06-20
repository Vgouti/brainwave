import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# Load datasets
@st.cache
def load_data():
    train_df = pd.read_csv(r'C:\Users\vaibh\OneDrive\Desktop\final project\finallymixed.csv')
    test_df = pd.read_csv(r'C:\Users\vaibh\OneDrive\Desktop\final project\trends.csv')
    return train_df.copy(), test_df.copy()

train_df, test_df = load_data()  # Load the data

# Modify the dataframes
train_df_modified = train_df.copy()  # Make a copy of the dataframe
# Perform modifications on train_df_modified

test_df_modified = test_df.copy()  # Make a copy of the dataframe
# Perform modifications on test_df_modified

# Use train_df_modified and test_df_modified in your Streamlit app

# Convert dates to datetime
train_df['Order Date'] = pd.to_datetime(train_df['Order Date'], format='%Y-%m-%d')
test_df['Day'] = pd.to_datetime(test_df['Day'], format='%d-%m-%Y')

# One-hot encode 'categories'
encoder = OneHotEncoder(sparse_output=False, drop='first')
train_encoded = encoder.fit_transform(train_df[['categories']])
train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(['categories']))
train_df = pd.concat([train_df.reset_index(drop=True), train_encoded_df], axis=1)

test_encoded = encoder.transform(test_df[['categories']])
test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(['categories']))
test_df = pd.concat([test_df.reset_index(drop=True), test_encoded_df], axis=1)

# UI
st.title("Sales Prediction App")
st.sidebar.title("Model Selection")
category = st.sidebar.selectbox("Select Category", train_df['categories'].unique())

# Filter the data for selected category
train_category_df = train_df[train_df['categories'] == category]
test_category_df = test_df[test_df['categories'] == category]

# Align columns
missing_cols = set(train_encoded_df.columns) - set(test_category_df.columns)
for c in missing_cols:
    test_category_df[c] = 0
test_category_df = test_category_df[['Day', 'Total'] + list(train_encoded_df.columns)]

# Features for training and testing
X_train = train_category_df[['Total'] + list(train_encoded_df.columns)]
y_train = train_category_df['Quantity']
X_test = test_category_df[['Total'] + list(train_encoded_df.columns)]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'SVR': SVR(),
    'XGBoost': XGBRegressor(),
    'LightGBM': LGBMRegressor(),
    'CatBoost': CatBoostRegressor(silent=True)
}

selected_model = st.sidebar.selectbox("Select Model", list(models.keys()))
model = models[selected_model]

# Train and predict
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Plot predictions
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(test_category_df['Day'], y_pred, label='Predicted Quantity', color='red')
ax.set_xlabel('Date')
ax.set_ylabel('Predicted Quantity')
ax.set_title(f'Predicted Quantity for {selected_model}')
ax.legend(loc='upper left')
ax.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Display MSE if actual quantities are available in test data
if 'Quantity' in test_category_df.columns:
    y_test = test_category_df['Quantity']
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error for {selected_model}: {mse}")
