from datetime import timedelta

import pandas
import pandas as pd
import numpy as np
import streamlit as st
import keras
import sklearn
from keras.models import Sequential
# from neuralprophet import NeuralProphet
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

device_select: str = st.sidebar.selectbox(
    "Which Device?",
    [f"Device {d}" for d in ("D", "I", "K", "S")]
)

device_id = device_select.removeprefix("Device ")


def load(dev: str) -> (pandas.DataFrame, Sequential):
    selected_model = keras.models.load_model(f"models/Device {dev}/")

    selected_data = pd.read_csv(f"project_datasets/data_{dev}.csv")

    return selected_data, selected_model


def prepare_data(df):
    shape = [-1] + [len(level) for level in df.columns.remove_unused_levels().levels]
    return df.values.reshape(shape)


def create_delayed_columns(series, times):
    cols = []
    column_index = []
    for time in times:
        cols.append(series.shift(-time))
        lag_fmt = "t+{time}" if time > 0 else "t{time}" if time < 0 else "t"
        column_index += [(lag_fmt.format(time=time), col_name)
                         for col_name in series.columns]
    df = pd.concat(cols, axis=1)
    df.columns = pd.MultiIndex.from_tuples(column_index)
    return df


def predict_df(df: pandas.DataFrame, model: Sequential, periods: int) -> pandas.DataFrame:
    df_copy = df.copy()
    df_copy = df_copy.fillna(method="ffill")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(df_copy)

    df_copy_scaled = df_copy
    df_copy_scaled["Temp"] = scaler.transform(df_copy)

    ## Look back (How many days in the past we check for prediction)
    previous_days = 20
    ## Which day we predict (1=Next Day)
    after_days = 1

    X = create_delayed_columns(df_copy_scaled, times=range(-previous_days + 1, 1)).iloc[previous_days:-after_days]
    X_train = X.loc[:"2020-12-28"]
    X_train_3D = prepare_data(X_train)
    steps_back = len(X_train_3D[1])

    #### Prediction
    ## Store forecast
    forecast = []

    first_eval_batch = data_scaled[-steps_back:]
    ## Shaping starting data
    current_batch = first_eval_batch.reshape((1, steps_back, 1))
    for i in range(periods):
        ## Get prediction value
        current_pred = model.predict(current_batch)[0]
        ## Store prediction value
        forecast.append(current_pred)
        ## Update the batch and drop the first value to keep the set length
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    forecast = scaler.inverse_transform(forecast)

    ## Create a date range from the last available datapoint
    last_timestamp = (df_copy.tail(1).index + timedelta(minutes=5))[0]
    forecast_index = pd.date_range(start=last_timestamp, periods=periods, freq='5T')

    ## Combine the prediction values and date range into a dataframe
    forecast_df = pd.DataFrame(data=forecast,
                               index=forecast_index,
                               columns=['Forecast'])

    forecast_df['Forecast'] = forecast_df['Forecast'].astype(float)

    return forecast_df

    # future = model.make_future_dataframe(df, periods=periods, n_historic_predictions=True)
    # forecast = model.predict(future)
    #
    # st.write(model.plot(forecast))


def do_device(data: pandas.DataFrame, model: Sequential, did: str):
    data["EventDt"] = pd.to_datetime(data["EventDt"])
    data = data.set_index(data["EventDt"])
    data = data.resample(rule="5T").mean()
    # data = data.rename(columns={"Temp": "y"})
    # data["ds"] = data.index

    min_date = data.index.min()
    max_date = data.index.max()

    st.title(f"Device {did}: Predictions")

    st.write(data)

    d = st.date_input("Latest date", value=max_date, min_value=min_date, max_value=max_date)

    p = st.number_input("Periods", value=50, min_value=50)

    print(d)

    df_selected: pandas.DataFrame = data[data.index <= str(d)]
    print(df_selected.info())
    df_excluded: pandas.DataFrame = data[data.index > str(d)]

    df_pred = predict_df(df_selected, model, p)

    fig: plt.Figure
    fig, ax = plt.subplots()

    df_selected.plot(kind='line', y="Temp", c="purple", ax=ax)
    df_excluded.plot(kind='line', y="Temp", c="blue", ax=ax)
    df_pred.plot(kind='line', y="Forecast", c="red", ax=ax)

    st.pyplot(fig)


d, m = load(device_id)

do_device(d, m, device_id)
