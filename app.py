import streamlit as st
from styles import overall_css
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet

# Set display options for pandas
pd.set_option('display.max_rows', 20)
pd.options.display.float_format = '{:.2f}'.format

# Function to read and display CSV data
def read_and_display_data(file_path):
    df = pd.read_csv(file_path)

    # Display the first 5 rows of the DataFrame
    st.subheader("First 5 Rows (Head) of the DataFrame")
    st.write(df.head(5))

    # Display the shape of the DataFrame
    st.subheader("Shape of the DataFrame")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Display the list of columns in the DataFrame
    st.subheader("Columns in the DataFrame")
    st.write(df.columns.tolist())

    # Display the data types of each column
    st.subheader("Data Types of Columns")
    st.write(df.dtypes)

    # Display summary statistics for numerical columns
    st.subheader("Summary Statistics (Numerical Columns)")
    st.write(df.describe())

    # Display the number of unique values per column
    st.subheader("Number of Unique Values per Column")
    st.write(df.nunique())

    # Rename necessary columns for Prophet
    df = df.rename(columns={'Dateofbill': 'ds', 'Quantity': 'y'})

    return df

# Function to display the highest selling drugs
def highest_selling_drugs(df):
    top_spec_drugs = df.groupby(['DrugName'])['y'].sum().reset_index()
    top_spec_drugs = top_spec_drugs.sort_values(by='y', ascending=False).head(20)

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    ax = sns.barplot(x='y', y='DrugName', data=top_spec_drugs, palette='bright')

    plt.xlabel('Quantity')
    plt.ylabel('Drug Name')
    plt.title('Top 20 Highest Selling Drug Names')

    for i, v in enumerate(top_spec_drugs['y']):
        ax.text(v + 0.5, i + 0.1, f"{round((v / top_spec_drugs['y'].sum()) * 100, 2)}%")

    st.pyplot(plt)

# Function to display the top products by quantity sold
def display_top_quantity_products(df):
    top_quantity = df.groupby(['DrugName'])['y'].sum().reset_index()
    top_quantity_products = top_quantity.sort_values(by='y', ascending=False).head(20)
    
    st.write(top_quantity_products)
    
    return top_quantity_products

# Forecasting function for weekly or monthly predictions
def Forecasting_top_sku(period: int, predictions_df: pd.DataFrame, freq: str, top_quantity_products, df):
    for top_quantity_product in top_quantity_products['DrugName']:
        drug_name = top_quantity_product
        new_df = df[df['DrugName'] == drug_name]
        s_df = new_df[['ds', 'y']].copy()
        s_model = Prophet()
        s_model.fit(s_df)
        future_dates = s_model.make_future_dataframe(periods=period, freq=freq)
        forecast = s_model.predict(future_dates)

        if freq == 'W':
            start_date = future_dates['ds'].iloc[-period]
            end_date = start_date + pd.DateOffset(days=6 * period)
            quantity_required = int(forecast['yhat'].iloc[-period:].sum())
        elif freq == 'M':
            start_date = future_dates['ds'].iloc[-period]
            end_date = start_date + pd.DateOffset(days=29 * period)
            quantity_required = int(forecast['yhat'].iloc[-period:].sum())

        predicted_row = pd.DataFrame({
            'DrugName': [drug_name],
            'QuantityRequired': [quantity_required],
            'StartDate': [start_date],
            'EndDate': [end_date]
        })
        predictions_df = pd.concat([predictions_df, predicted_row], ignore_index=True)

    return predictions_df, s_model, forecast

# Plotting the forecast results as bar charts
def plot_forecast_barchart(predictions_df: pd.DataFrame, title: str):
    if {'DrugName', 'QuantityRequired'}.issubset(predictions_df.columns):
        plt.figure(figsize=(10, 6))
        plt.bar(predictions_df['DrugName'], predictions_df['QuantityRequired'], color='skyblue')
        plt.title(title, fontsize=16)
        plt.xlabel('Drug Name', fontsize=14)
        plt.ylabel('Quantity Required', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt)
    else:
        st.error("The DataFrame must contain 'DrugName' and 'QuantityRequired' columns.")

# Function for weekly forecast
def weekly_forecast(top_quantity_products, df, period):
    st.write(f"Forecasting for {period} week(s)")

    # Create empty DataFrame for predictions
    predictions_df = pd.DataFrame(columns=['DrugName', 'StartDate', 'EndDate', 'QuantityRequired'])
    predictions_df, _, _ = Forecasting_top_sku(period, predictions_df, 'W', top_quantity_products, df)
    plot_forecast_barchart(predictions_df, f"Weekly Forecast for Top Products (Period: {period} weeks)")
    st.write(predictions_df)

# Function for monthly forecast
def monthly_forecast(top_quantity_products, df, period):
    st.write(f"Forecasting for {period} month(s)")

    # Create empty DataFrame for predictions
    predictions_df = pd.DataFrame(columns=['DrugName', 'StartDate', 'EndDate', 'QuantityRequired'])
    predictions_df, _, _ = Forecasting_top_sku(period, predictions_df, 'M', top_quantity_products, df)
    plot_forecast_barchart(predictions_df, f"Monthly Forecast for Top Products (Period: {period} months)")
    st.write(predictions_df)

# Sidebar for file upload
def sidebar():
    st.sidebar.title('Drug Sales Forecast')
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type='csv')
    return uploaded_file

# Main function to show the app
def show():
    st.markdown(overall_css, unsafe_allow_html=True)
    st.markdown("<h1>Drug Sales Data Analysis and Forecasting</h1>", unsafe_allow_html=True)

    uploaded_file = sidebar()

    if uploaded_file is not None:
        df = read_and_display_data(uploaded_file)

        st.subheader("Top Selling Drugs")
        highest_selling_drugs(df)

        st.subheader("Top Products by Quantity Sold")
        top_quantity_products = display_top_quantity_products(df)

        # Input for forecast period
        forecast_period = st.sidebar.slider("Select Forecast Period", 1, 12, 1)

        # Create buttons for weekly and monthly forecast
        weekly_button = st.sidebar.button("Weekly Forecast")
        monthly_button = st.sidebar.button("Monthly Forecast")

        if weekly_button:
            for period in range(1, forecast_period + 1):
                weekly_forecast(top_quantity_products, df, period)  # Pass the period argument here

        if monthly_button:
            for period in range(1, forecast_period + 1):
                monthly_forecast(top_quantity_products, df, period)  # Pass the period argument here

    else:
        st.warning("Please upload a CSV file.")

if __name__ == "__main__":
    show()
