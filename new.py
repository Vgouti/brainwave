import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# Load datasets
@st.cache_data
def load_data():
    orders_df = pd.read_csv(r'C:\Users\vaibh\OneDrive\Desktop\final project\finallymixed.csv')
    trends_df = pd.read_csv(r'C:\Users\vaibh\OneDrive\Desktop\final project\trends.csv')
    return orders_df, trends_df

orders_df, trends_df = load_data()

# Convert dates to datetime
orders_df['Order Date'] = pd.to_datetime(orders_df['Order Date'], format='%Y-%m-%d')
trends_df['Day'] = pd.to_datetime(trends_df['Day'], format='%d-%m-%Y')

# Basic dashboard
st.title("Sales Prediction Dashboard")
option = st.sidebar.selectbox(
    'Select Option:',
    ('Basic Dashboard', 'Quantity Prediction', 'Future Prediction')
)

if option == 'Basic Dashboard':
    st.subheader("Sales Comparison by Category")
    fig_bar = px.bar(trends_df, x='Day', y='Total', title='Total Sales by Category')
    st.plotly_chart(fig_bar)

    #Revenue Comparison by Category
    #st.subheader("Revenue Comparison by Category")
    #orders_df['Revenue'] = orders_df['actual_price'] * orders_df['Quantity']
    #fig_bar_revenue = px.bar(orders_df, x='categories', y='Revenue', title='Total Revenue by Category')
    #st.plotly_chart(fig_bar_revenue)

    #Distribution of Sales by Category
    #st.subheader("Distribution of Sales by Category")
    #fig_pie = px.pie(orders_df, names='categories', title='Distribution of Sales by Category')
    
    
    #st.subheader("Sales Heatmap by State")
    #fig_heatmap = px.density_mapbox(orders_df, lat='Latitude', lon='Longitude', z='Quantity', radius=10,
                                    #center=dict(lat=20.5937, lon=78.9629), zoom=3,
                                    #mapbox_style="stamen-terrain", title='Sales Heatmap by State')
    #st.plotly_chart(fig_heatmap)

elif option == 'Quantity Prediction':
    st.subheader("Quantity Prediction using Linear Regression and Random Forest")
    orders_df['Order Date'] = pd.to_datetime(orders_df['Order Date'], format='%Y-%m-%d')
    trends_df['Day'] = pd.to_datetime(trends_df['Day'], format='%d-%m-%Y')

    # One-hot encode 'categories'
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    train_encoded = encoder.fit_transform(orders_df[['categories']])
    train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(['categories']))
    orders_df = pd.concat([orders_df.reset_index(drop=True), train_encoded_df], axis=1)

    test_encoded = encoder.transform(trends_df[['categories']])
    test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(['categories']))
    trends_df = pd.concat([trends_df.reset_index(drop=True), test_encoded_df], axis=1)

    # UI
    st.title("Sales Prediction App")
    st.sidebar.title("Model Selection")
    category = st.sidebar.selectbox("Select Category", orders_df['categories'].unique())

    # Filter the data for selected category
    train_category_df = orders_df[orders_df['categories'] == category]
    test_category_df = trends_df[trends_df['categories'] == category]

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

elif option == 'Future Prediction':
    st.subheader("Future Sales Prediction using Prophet")
    # Filter data for a specific category
    category = st.selectbox("Select Category", orders_df['categories'].unique())
    category_data = orders_df[orders_df['categories'] == category]

    # Check if there is enough data for modeling
    if category_data.shape[0] < 2:
        raise Exception(f"Insufficient data for modeling in category: {category}")

    # Convert 'Order Date' to datetime
    category_data['Order Date'] = pd.to_datetime(category_data['Order Date'])

    # Prepare the data for Prophet
    prophet_data = category_data[['Order Date', 'Quantity']].rename(columns={'Order Date': 'ds', 'Quantity': 'y'})

    # Resample to monthly data
    prophet_data = prophet_data.set_index('ds').resample('ME').sum().reset_index()

    # Fit the Prophet model
    model = Prophet()
    model.fit(prophet_data)

    # Create a DataFrame to hold future dates for predictions
    future_dates = model.make_future_dataframe(periods=12, freq='M')  # Forecast for the next 12 months

    # Make predictions
    forecast = model.predict(future_dates)

    # Plot the forecast
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(prophet_data['ds'], prophet_data['y'], 'bo-', label='Actual Sales Data')  # 'bo-' means blue line with dots
    forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast_data.set_index('ds', inplace=True)
    ax.plot(forecast_data.index, forecast_data['yhat'], 'orange', label='Forecasted Sales')
    ax.fill_between(forecast_data.index, forecast_data['yhat_lower'], forecast_data['yhat_upper'], color='orange', alpha=0.2, label='Confidence Interval')
    ax.set_title(f'Sales Quantity Forecast for {category}', fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Sales Quantity', fontsize=14)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

