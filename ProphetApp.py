import streamlit as st
import pandas as pd
import pickle
import datetime

pickle_in = open("prophet_model.pkl", "rb")
model_fit = pickle.load(pickle_in)

def welcome():
    return "Welcome all"

def predict_oil_price(diff):
    future = model_fit.make_future_dataframe(periods=diff)
    forecast = model_fit.predict(future)
    prediction = forecast.tail(1)['yhat'].item()
    return prediction

def main():
    st.title("Crude Oil Price Prediction")
    html_temp = """
    <div style="background-color: tomato; padding: 10px">
    <h2 style="color: white; text-align: center;">Model Deployment: Timeseries Forecasting</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    s = datetime.date(year=2023, month=6, day=12)
    e = st.date_input("Enter the ending Date to Predict the Oil Prices")
    diff = (e - s).days + 1
    
    if st.button("PREDICT"):
        index_future_dates = pd.date_range(start=s, end=e)
        final_forecast = predict_oil_price(diff)
        pred = pd.DataFrame(final_forecast, index=index_future_dates.rename("Date"), columns=["Price"])
        
        st.dataframe(pred)
        st.line_chart(pred)
    
if __name__ == '__main__':
    main()
