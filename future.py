import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go

# Load datasets
@st.cache_data
def load_data():
    train_df = pd.read_csv('finallymixed.csv')
    test_df = pd.read_csv('trends.csv')
    return train_df, test_df

train_df, test_df = load_data()

# Convert dates to datetime
train_df['Order Date'] = pd.to_datetime(train_df['Order Date'], format='%Y-%m-%d')
test_df['Day'] = pd.to_datetime(test_df['Day'], format='%d-%m-%Y')

# Basic Dashboard with Enhanced Visualizations
def display_dashboard(selected_category):
    st.title(f"Basic Dashboard for {selected_category}")

    # Filter the data for selected category
    train_category_df = train_df[train_df['categories'] == selected_category]
    trends_category_df = test_df[test_df['categories'] == selected_category]

    # Sales Trends Line Graph
    st.subheader("Sales Trends Comparison")
    fig_line = px.line(trends_category_df, x='Day', y='Total', color='categories', title='Sales Trends Over Time')
    st.plotly_chart(fig_line)

    # Heatmap for Sales by State
    st.subheader("Sales Heatmap by State")
    state_sales = train_category_df.groupby('Location')['Quantity'].sum().reset_index()
    fig_map = px.choropleth(state_sales, locations='Location', locationmode='country names', color='Quantity', scope="asia", title='Sales Heatmap by State')
    st.plotly_chart(fig_map)

# Sidebar Menu for Navigation
st.sidebar.title("Navigation")
menu = st.sidebar.selectbox("Menu", ["Basic Dashboard", "Quantity Prediction", "Future Prediction"])

if menu == "Basic Dashboard":
    st.subheader("Sales Comparison by Category")
    fig_bar = px.bar(train_df, x='categories', y='Quantity', title='Total Sales by Category')
    st.plotly_chart(fig_bar)

    # Revenue Comparison by Category
    st.subheader("Revenue Comparison by Category")
    train_df['Revenue'] = train_df['actual_price'] * train_df['Quantity']
    fig_bar_revenue = px.bar(train_df, x='categories', y='Revenue', title='Total Revenue by Category')
    st.plotly_chart(fig_bar_revenue)

    # Pie Chart for Sales Distribution by Category
    st.subheader("Distribution of Sales by Category")
    fig_pie = px.pie(train_df, names='categories', title='Distribution of Sales by Category')
    st.plotly_chart(fig_pie)

    selected_category = st.sidebar.selectbox("Select Category for Dashboard", train_df['categories'].unique())
    display_dashboard(selected_category)

elif menu == "Quantity Prediction":
    st.title("Quantity Prediction")

    # One-hot encode 'categories'
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    train_encoded = encoder.fit_transform(train_df[['categories']])
    train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(['categories']))
    train_df = pd.concat([train_df.reset_index(drop=True), train_encoded_df], axis=1)

    test_encoded = encoder.transform(test_df[['categories']])
    test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(['categories']))
    test_df = pd.concat([test_df.reset_index(drop=True), test_encoded_df], axis=1)

    # Filter the data for selected category
    category = st.sidebar.selectbox("Select Category", train_df['categories'].unique())
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
    }

    selected_model = st.sidebar.selectbox("Select Model", list(models.keys()))
    model = models[selected_model]

    # Train and predict
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Plot predictions with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_category_df['Day'], y=y_pred, mode='lines+markers', name='Predicted Quantity', line=dict(color='red')))
    fig.update_layout(title=f'Predicted Quantity for {selected_model}', xaxis_title='Date', yaxis_title='Predicted Quantity')
    st.plotly_chart(fig)

    # Display MSE if actual quantities are available in test data
    if 'Quantity' in test_category_df.columns:
        y_test = test_category_df['Quantity']
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error for {selected_model}: {mse}")

elif menu == "Future Prediction":
    st.title("Future Prediction with Prophet")

    # Prepare data for Prophet
    prophet_data = train_df.rename(columns={'Order Date': 'ds', 'Quantity': 'y'})
    prophet_model = Prophet()
    prophet_model.fit(prophet_data)

    # Make future predictions
    future = prophet_model.make_future_dataframe(periods=365)
    forecast = prophet_model.predict(future)

    # Filter future predictions
    future_predictions = forecast[forecast['ds'] >= '2024-01-01']

    # Prepare actual future values from test data
    actual_future = test_df[['Day', 'Total']].rename(columns={'Day': 'ds', 'Total': 'y'})

    # Ensure both ds columns are of datetime type
    future_predictions['ds'] = pd.to_datetime(future_predictions['ds'])
    actual_future['ds'] = pd.to_datetime(actual_future['ds'])

    # Merge actual values from test data with predictions
    merged = pd.merge(future_predictions, actual_future, on='ds', how='inner')

    # Plot predictions with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prophet_data['ds'], y=prophet_data['y'], mode='lines+markers', name='Actual Sales Data', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=future_predictions['ds'], y=future_predictions['yhat'], mode='lines', name='Forecasted Sales', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=future_predictions['ds'], y=future_predictions['yhat_lower'], fill='tonexty', mode='lines', line=dict(color='orange', width=0), name='Confidence Interval Lower Bound'))
    fig.add_trace(go.Scatter(x=future_predictions['ds'], y=future_predictions['yhat_upper'], fill='tonexty', mode='lines', line=dict(color='orange', width=0), name='Confidence Interval Upper Bound'))
    fig.update_layout(title=f'Sales Quantity Forecast with Prophet', xaxis_title='Date', yaxis_title='Sales Quantity')
    st.plotly_chart(fig)
