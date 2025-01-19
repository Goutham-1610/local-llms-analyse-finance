import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from datetime import datetime

# Title
st.title("Financial Dashboard")

# Load data
file_path = "transactions_2022_2023_categorized.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    st.error("File not found. Please check the file path and try again.")
    st.stop()

# Add year and month columns
df['Year'] = pd.to_datetime(df['Date']).dt.year
df['Month'] = pd.to_datetime(df['Date']).dt.month
df['Month Name'] = pd.to_datetime(df['Date']).dt.strftime("%b")

# Remove unnecessary columns
if 'Transaction' in df.columns and 'Transaction vs category' in df.columns:
    df = df.drop(columns=['Transaction', 'Transaction vs category'])

# Adjust categories for income rows
df['Category'] = np.where(df['Expense/Income'] == 'Income', df['Name / Description'], df['Category'])

# Function to create pie chart
def make_pie_chart(df, year, label):
    sub_df = df[(df['Expense/Income'] == label) & (df['Year'] == year)]
    color_scale = px.colors.qualitative.Set2
    
    pie_fig = px.pie(sub_df, values='Amount (EUR)', names='Category', color_discrete_sequence=color_scale)
    pie_fig.update_traces(textposition='inside', direction='clockwise', hole=0.3, textinfo="label+percent")

    total_expense = df[(df['Expense/Income'] == 'Expense') & (df['Year'] == year)]['Amount (EUR)'].sum()
    total_income = df[(df['Expense/Income'] == 'Income') & (df['Year'] == year)]['Amount (EUR)'].sum()
    
    if label == 'Expense':
        total_text = "€ " + str(round(total_expense))
        saving_rate = round((total_income - total_expense) / total_income * 100)
        saving_rate_text = ": Saving rate " + str(saving_rate) + "%"
    else:
        total_text = "€ " + str(round(total_income))
        saving_rate_text = ""

    pie_fig.update_layout(
        uniformtext_minsize=10, 
        uniformtext_mode='hide',
        title=dict(text=f"{label} Breakdown {year} {saving_rate_text}"),
        annotations=[
            dict(
                text=total_text, 
                x=0.5, y=0.5, font_size=12,
                showarrow=False
            )
        ]
    )
    return pie_fig

# Function to create monthly bar chart
def make_monthly_bar_chart(df, year, label):
    df = df[(df['Expense/Income'] == label) & (df['Year'] == year)]
    total_by_month = (df.groupby(['Month', 'Month Name'])['Amount (EUR)'].sum()
                      .reset_index()
                      .sort_values(by='Month')
                      .reset_index(drop=True))
    
    color_scale = px.colors.sequential.YlGn if label == "Income" else px.colors.sequential.OrRd
    
    bar_fig = px.bar(
        total_by_month, 
        x='Month Name', 
        y='Amount (EUR)', 
        text_auto='.2s', 
        title=f"{label} per month", 
        color='Amount (EUR)', 
        color_continuous_scale=color_scale
    )
    return bar_fig

# Display full table of categorized transactions
st.subheader("Categorized Transactions")
st.dataframe(df, use_container_width=True)  # Displays the entire table with scrollable functionality

# Select Year
year = st.selectbox("Select Year", df['Year'].unique())

# Pie Chart for Income and Expense
st.subheader("Pie Charts")
st.plotly_chart(make_pie_chart(df, year, 'Income'), use_container_width=True)
st.plotly_chart(make_pie_chart(df, year, 'Expense'), use_container_width=True)

# Bar Chart for Income and Expense
st.subheader("Bar Charts")
st.plotly_chart(make_monthly_bar_chart(df, year, 'Income'), use_container_width=True)
st.plotly_chart(make_monthly_bar_chart(df, year, 'Expense'), use_container_width=True)

# Download button for categorized data
st.subheader("Download Categorized Data")
st.download_button(
    label="Download CSV",
    data=df.to_csv(index=False),
    file_name="categorized_transactions.csv",
    mime="text/csv"
)

# Bill Payment Reminders
st.subheader("Upcoming Bill Payment Reminders")
today = datetime.now()
upcoming_bills = df[(df['Category'].str.contains('Bills|Utilities', case=False, na=False)) &
                    (pd.to_datetime(df['Date']) >= today)]
if not upcoming_bills.empty:
    upcoming_bills = upcoming_bills[['Date', 'Category', 'Amount (EUR)', 'Description']]
    st.write("You have the following upcoming bills:")
    st.dataframe(upcoming_bills)
else:
    st.write("No upcoming bills found.")

# Chatbot Integration
st.subheader("Financial Chatbot")
user_query = st.text_input("Ask your financial question:")

if user_query:
    # Prepare dashboard context
    expense_summary = df[df['Expense/Income'] == 'Expense'].groupby('Category')['Amount (EUR)'].sum().to_dict()
    income_summary = df[df['Expense/Income'] == 'Income'].groupby('Category')['Amount (EUR)'].sum().to_dict()

    dashboard_context = f"""
    Expense Summary: {expense_summary}
    Income Summary: {income_summary}
    Current Year Selected: {year}.
    """

    # Preprompt for financial questions
    preprompt = (
        "You are a financial assistant AI. Answer only questions related to personal finance, "
        "budgeting, saving, and financial planning. Use the following dashboard data to answer:\n\n"
        f"{dashboard_context}"
    )
    
    # LLM Studio Chat Completions API endpoint
    llm_endpoint = "http://localhost:1234/v1/chat/completions"

    # Request payload
    payload = {
        "model": "default",  # Replace 'default' with the actual model name if needed
        "messages": [
            {"role": "system", "content": preprompt},
            {"role": "user", "content": user_query}
        ],
        "max_tokens": 150,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    # Send request to LLM
    try:
        response = requests.post(llm_endpoint, json=payload)
        if response.status_code == 200:
            bot_response = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            st.write(f"Chatbot: {bot_response}")
        else:
            st.error(f"Failed to fetch response from LLM. Error code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to LLM: {e}")
