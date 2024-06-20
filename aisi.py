import pandas as pd
import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import folium_static
import json
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from prophet import Prophet
import plotly.graph_objects as go

@st.cache_data
def load_data():
    train_df = pd.read_csv(r'C:\Users\vaibh\OneDrive\Desktop\final project\stateidmixed.csv')
    testing_df = pd.read_csv(r"C:\Users\vaibh\OneDrive\Desktop\final project\trends.csv")
    return train_df, testing_df

train_df, testing_df = load_data()

train_df['Order Date'] = pd.to_datetime(train_df['Order Date'], format='%Y-%m-%d')
testing_df['Day'] = pd.to_datetime(testing_df['Day'], format='%d-%m-%Y')
train_df['Month'] = train_df['Order Date'].dt.to_period('M').astype(str)

st.sidebar.title("Navigation")
menu = st.sidebar.selectbox("Menu", ["Basic Dashboard", "Category-Specific Dashboard", "Quantity Prediction", "Future Prediction"])

if menu == "Basic Dashboard":
    st.subheader("Date and Quantity Line Graph by Category")
    # Aggregate data by month
    monthly_data = train_df.groupby('Month')['Quantity'].sum().reset_index()

    # Plot aggregated data
    fig_line = px.line(monthly_data, x='Month', y='Quantity', 
                    title='Total Quantity Sold Over Time',
                    labels={'Month': 'Month', 'Quantity': 'Total Quantity Sold'},
                    template='plotly_dark')
    fig_line.update_layout(
        xaxis_title='Month',
        yaxis_title='Total Quantity Sold',
        hovermode='x unified'
    )

    st.plotly_chart(fig_line)


    st.subheader("Pie Chart for Sales in Each Category")
    category_sales = train_df.groupby('categories')['Quantity'].sum().reset_index()
    fig_pie = px.pie(category_sales, names='categories', values='Quantity', title='Distribution of Sales by Category',
                     hover_data={'Quantity': True})
    st.plotly_chart(fig_pie)
    
    st.subheader("Revenue from Each Category Bar Chart")
    train_df['Revenue'] = train_df['actual_price'] * train_df['Quantity']
    fig_bar_revenue = px.bar(train_df, x='categories', y='Revenue', title='Total Revenue by Category',
                            hover_data={'Revenue': True, 'categories': False})
    st.plotly_chart(fig_bar_revenue)

if menu == "Category-Specific Dashboard":
    selected_category = st.sidebar.selectbox("Select Category", train_df['categories'].unique())
    category_df = train_df[train_df['categories'] == selected_category]

    st.subheader(f"Heatmap for Total Quantity Sold in Each State")
    state_quantity = category_df.groupby('state_id').agg({'Quantity': 'sum'}).reset_index()
    m = folium.Map(location=[23.47, 77.94], tiles='CartoDB positron', zoom_start=5, attr="My Data attribution")
    geojson_path = r"C:\Users\vaibh\OneDrive\Desktop\final project\india_telengana.geojson"

    # Create a dictionary to map state_id to quantity
    quantity_dict = state_quantity.set_index('state_id')['Quantity'].to_dict()

    # Load and process the GeoJSON file
    with open(geojson_path) as f:
        geojson_data = json.load(f)

    # Add quantity and state name to GeoJSON properties
    for feature in geojson_data['features']:
        state_id = feature['properties']['ID_1']
        quantity = quantity_dict.get(state_id, 'No data available')
        feature['properties']['Quantity'] = quantity

    # Filter out features with missing geometries
    geojson_data['features'] = [feature for feature in geojson_data['features'] if feature['geometry']]

    # Create the choropleth map
    choropleth = folium.Choropleth(
        geo_data=geojson_data,
        name="choropleth",
        data=state_quantity,
        columns=["state_id", "Quantity"],
        key_on="feature.properties.ID_1",
        fill_color="YlGnBu",  # Change to a ColorBrewer bluish color scheme
        fill_opacity=1,
        line_opacity=0,
        legend_name="Quantity"
    ).add_to(m)

    # Add tooltips
    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(['NAME_1', 'Quantity'], aliases=['State: ', 'Quantity: '])
    )

    folium_static(m, width=800, height=600)

    st.subheader(f"Monthly Sales in {selected_category}")
    monthly_sales = category_df.groupby('Month')['Quantity'].sum().reset_index()
    fig_monthly_sales = px.bar(monthly_sales, x='Month', y='Quantity', title=f'Monthly Sales in {selected_category}',
                            hover_data={'Month': False, 'Quantity': True},
                            color_discrete_sequence=['#636EFA'])  # Custom color
    fig_monthly_sales.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.8)  # Add border to bars
    fig_monthly_sales.update_layout(yaxis_title='Quantity Sold', xaxis_title='Month', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_monthly_sales)

    st.subheader(f"Sales by Location in {selected_category}")
    location_sales = category_df.groupby('Location')['Quantity'].sum().reset_index()
    fig_location_sales = px.bar(location_sales, x='Location', y='Quantity',
                                title=f'Sales by Location in {selected_category}',
                                hover_data={'Location': False, 'Quantity': True},
                                color_discrete_sequence=['#EF553B'])  # Custom color
    fig_location_sales.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.8)  # Add border to bars
    fig_location_sales.update_layout(yaxis_title='Quantity Sold', xaxis_title='Location', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_location_sales)

elif menu == 'Quantity Prediction':
    st.subheader("Quantity Prediction using Linear Regression and Random Forest")

    # One-hot encode 'categories'
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    train_encoded = encoder.fit_transform(train_df[['categories']])
    train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(['categories']))
    train_df = pd.concat([train_df.reset_index(drop=True), train_encoded_df], axis=1)

    test_encoded = encoder.transform(testing_df[['categories']])
    test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(['categories']))
    testing_df = pd.concat([testing_df.reset_index(drop=True), test_encoded_df], axis=1)

    # UI
    st.title("Sales Prediction App")
    st.sidebar.title("Model Selection")
    category = st.sidebar.selectbox("Select Category", train_df['categories'].unique())

    # Filter the data for selected category
    train_category_df = train_df[train_df['categories'] == category]
    test_category_df = testing_df[testing_df['categories'] == category]

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

    # Prepare predictions DataFrame
    predictions_df = pd.DataFrame({'Date': test_category_df['Day'], 'Predicted Quantity': y_pred})

    # Plot predictions as line graph
    fig_line = px.line(predictions_df, x='Date', y='Predicted Quantity', title=f'Predicted Quantity for {selected_model}',
                  labels={'Date': 'Date', 'Predicted Quantity': 'Predicted Quantity'})
    fig_line.update_layout(hovermode='x unified')
    st.plotly_chart(fig_line)

    # Plot predictions as bar graph
    fig_bar = px.bar(predictions_df, x='Date', y='Predicted Quantity', title=f'Predicted Quantity for {selected_model}',
                     labels={'Date': 'Date', 'Predicted Quantity': 'Predicted Quantity'})
    fig_bar.update_layout(hovermode='x unified')
    st.plotly_chart(fig_bar)

    # Display predictions DataFrame and download option
    st.write(predictions_df)
    csv = predictions_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions as CSV", data=csv, file_name='predictions.csv', mime='text/csv')

    # Add option to write timeline
    timeline_text = st.text_input("Write Timeline for the Predictions")
    if timeline_text:
        st.markdown(f"**Timeline:** {timeline_text}")
elif menu == "Future Prediction":
    category = st.selectbox("Select Category", train_df['categories'].unique())
    category_data = train_df[train_df['categories'] == category]

    # Check if there is enough data for modeling
    if category_data.shape[0] < 2:
        raise Exception(f"Insufficient data for modeling in category: {category}")

    # Convert 'Order Date' to datetime
    category_data['Order Date'] = pd.to_datetime(category_data['Order Date'])

    # Prepare the data for Prophet
    prophet_data = category_data[['Order Date', 'Quantity']].rename(columns={'Order Date': 'ds', 'Quantity': 'y'})

    # Resample to monthly data
    prophet_data = prophet_data.set_index('ds').resample('M').sum().reset_index()

    # Fit the Prophet model
    model = Prophet()
    model.fit(prophet_data)

    # Create a DataFrame to hold future dates for predictions
    future_dates = model.make_future_dataframe(periods=12, freq='M')  # Forecast for the next 12 months

    # Make predictions
    forecast = model.predict(future_dates)
    forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    # Concatenate actual and forecast data
    actual_and_forecast = pd.concat([prophet_data.set_index('ds'), forecast_data.set_index('ds')], axis=1).reset_index()

    # Plot predictions with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prophet_data['ds'], y=prophet_data['y'], mode='lines+markers', name='Actual Sales Data', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['yhat'], mode='lines', name='Forecasted Sales', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['yhat_lower'], fill='tonexty', mode='lines', line=dict(color='orange', width=0), name='Confidence Interval Lower Bound'))
    fig.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['yhat_upper'], fill='tonexty', mode='lines', line=dict(color='orange', width=0), name='Confidence Interval Upper Bound'))
    fig.update_layout(title=f'Sales Quantity Forecast with Prophet', xaxis_title='Date', yaxis_title='Sales Quantity')
    st.plotly_chart(fig)

    # Display future predictions DataFrame and download option
    st.write(forecast_data)
    csv = actual_and_forecast.to_csv(index=False).encode('utf-8')
    st.download_button("Download Future Predictions as CSV", data=csv, file_name='forecast_data.csv', mime='text/csv')

    # Add option to write timeline
    timeline_text = st.text_input("Write Timeline for the Predictions")
    if timeline_text:
        st.markdown(f"**Timeline:** {timeline_text}")

