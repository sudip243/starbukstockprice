import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('Starbucks_stock_history.csv')
df['MA20'] = df['Close'].rolling(window=20).mean()

st.title("📈 Starbucks Stock Analysis")

st.subheader("Raw Data")
st.dataframe(df.head())

# Sidebar feature selection
st.sidebar.title("Features Selection")
selected_features = st.sidebar.multiselect(
    'Choose input features:', ['Open', 'High', 'Low', 'Volume'], default=['Open', 'High']
)

# Plot section
st.subheader("📊 Close Price & Moving Average")
fig, ax = plt.subplots()
ax.plot(df['Close'], label='Close Price')
ax.plot(df['MA20'], label='20-Day MA', color='orange')
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Model training
if st.button('Train Model'):
    df_clean = df.fillna(0)
    X = df_clean[selected_features]
    y = df_clean['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    st.success(f"Model trained. R² Score: {score:.2f}")
    st.write("Model Coefficients:", model.coef_)
    st.write("Model Intercept:", model.intercept_)
