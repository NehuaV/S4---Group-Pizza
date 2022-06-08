from datetime import timedelta, date, datetime

import pandas
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import keras
import sklearn
from keras.models import Sequential

# from neuralprophet import NeuralProphet
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

device_select: str = st.sidebar.selectbox(
    "Which Device?", [f"Device {d}" for d in ("K", "S", "I", "D")]
)

limits = {
    "K": {
        "upper": -20,
        "lower": None,
    },
    "S": {
        "upper": 4,
        "lower": 0,
    },
    "I": {
        "upper": 25,
        "lower": 8,
    },
    "D": None,
}

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
        column_index += [
            (lag_fmt.format(time=time), col_name) for col_name in series.columns
        ]
    df = pd.concat(cols, axis=1)
    df.columns = pd.MultiIndex.from_tuples(column_index)
    return df


def predict_df(
    df: pandas.DataFrame, model: Sequential, periods: int
) -> pandas.DataFrame:
    df_copy = df.copy()
    df_copy = df_copy.fillna(method="ffill")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(df_copy)

    df_copy_scaled = df_copy
    df_copy_scaled["Temp"] = scaler.transform(df_copy)

    ## Look back (How many minutes in the past we check for prediction)
    previous_minutes = 200
    ## Which minute we predict (1=Next Minute)
    after_minute = 1

    X = create_delayed_columns(
        df_copy_scaled, times=range(-previous_minutes + 1, 1)
    ).iloc[previous_minutes:-after_minute]
    X_train = X.loc[:"2020-12-28"]
    X_train_3D = prepare_data(X_train)
    steps_back = len(X_train_3D[1])

    #### Prediction
    ## Store forecast
    forecast = []

    first_eval_batch = data_scaled[-steps_back:]
    ## Shaping starting data
    current_batch = first_eval_batch.reshape((1, steps_back, 1))

    prog = st.progress(0)

    for i in range(periods):
        ## Get prediction value
        current_pred = model.predict(current_batch)[0]
        ## Store prediction value
        forecast.append(current_pred)
        ## Update the batch and drop the first value to keep the set length
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

        if i != 0:
            prog.progress((i / periods))

    prog.progress(1.0)

    forecast = scaler.inverse_transform(forecast)

    ## Create a date range from the last available datapoint
    last_timestamp = (df_copy.tail(1).index + timedelta(minutes=5))[0]
    forecast_index = pd.date_range(start=last_timestamp, periods=periods, freq="5T")

    ## Combine the prediction values and date range into a dataframe
    forecast_df = pd.DataFrame(
        data=forecast, index=forecast_index, columns=["Forecast"]
    )

    forecast_df["Forecast"] = forecast_df["Forecast"].astype(float)

    # last_entry = df.iloc[-1].to_frame().rename({"Temp": "Forecast"})
    #
    # print((last_entry, last_entry.name, type(last_entry)))
    # # last_entry["Forecast"] = last_entry["Temp"]
    # # del last_entry["Temp"]
    # #
    # # forecast_df.append(last_entry)

    forecast_df = pd.concat(
        [df.iloc[-1:].rename(columns={"Temp": "Forecast"}), forecast_df]
    )

    return forecast_df

    # future = model.make_future_dataframe(df, periods=periods, n_historic_predictions=True)
    # forecast = model.predict(future)
    #
    # st.write(model.plot(forecast))


DAY = "Day"
WEEK = "Week"
MONTH = "Month"


def plot_zoom(
    end_date: date,
    zoom_select: str,
    df_selected: pandas.DataFrame,
    df_pred: pandas.DataFrame,
):
    if zoom_select == DAY:
        return pandas.concat(
            [
                df_selected[df_selected.index >= (end_date - timedelta(days=1))],
                df_pred,
            ]
        )
    elif zoom_select == WEEK:
        return pandas.concat(
            [
                df_selected[df_selected.index >= (end_date - timedelta(weeks=1))],
                df_pred,
            ]
        )
    elif zoom_select == MONTH:
        return pandas.concat(
            [
                df_selected[df_selected.index >= (end_date - timedelta(weeks=4))],
                df_pred,
            ]
        )
    else:
        return None


FORECAST_STEP = timedelta(minutes=5)


def do_device(data: pandas.DataFrame, model: Sequential, did: str):
    data["EventDt"] = pd.to_datetime(data["EventDt"])
    data = data.set_index(data["EventDt"])
    data = data.resample(rule="5T").mean()
    # data = data.rename(columns={"Temp": "y"})
    # data["ds"] = data.index

    min_date = data.index.min().to_pydatetime()
    print(min_date)
    max_date = data.index.max().to_pydatetime()
    print(max_date)

    st.title(f"Device {did}: Predictions")

    start_date, end_date = st.slider(
        "Prediction Date Range", min_date, max_date, (min_date, max_date)
    )

    until: datetime = st.slider(
        "Forecast Until",
        end_date + FORECAST_STEP,
        end_date + timedelta(weeks=2),
        end_date + (FORECAST_STEP * 100),
        timedelta(minutes=5),
    )
    with_alarm = st.checkbox("Enable alarm")
    zoom_select: str = st.selectbox(
        "Plot Zoom History", [DAY, WEEK, MONTH], index=0
    )

    periods = int((until - end_date) / timedelta(minutes=5))

    print((start_date, end_date))
    print(((end_date - until), periods))
    print(with_alarm)
    print(zoom_select)

    # df_selected: pandas.DataFrame = data[start_date:end_date]
    df_selected: pandas.DataFrame = data[
        (data.index >= start_date) & (data.index <= end_date)
    ]
    print(df_selected.info())
    df_excluded: pandas.DataFrame = data[data.index > str(end_date)]

    df_pred = predict_df(df_selected, model, periods)

    # print("Pred" + df_pred)

    # fig: plt.Figure
    # fig, ax = plt.subplots()

    df_combined = plot_zoom(end_date, zoom_select, df_selected, df_pred)
    print(df_combined.info())

    # df_selected.plot(kind='line', y="Temp", c="purple", ax=ax)
    # df_excluded.plot(kind='line', y="Temp", c="blue", ax=ax)
    # df_pred.plot(kind='line', y="Forecast", c="red", ax=ax)

    fig = px.line(title=f"Device {device_id} forecast")
    fig.add_trace(go.Scatter(x=df_combined.index, y=df_combined.Temp, name="Data"))
    fig.add_trace(
        go.Scatter(x=df_combined.index, y=df_combined.Forecast, name="Forecast")
    )
    fig.update_layout(hovermode="x")

    alarm_triggered = False

    if with_alarm:
        limit = limits[device_id]

        upper = None
        lower = None

        if limit is None:
            st.text(f"Note: Device {device_id} does not have limits")
        else:
            if limit["upper"] is None:
                st.text(f"Note: Device {device_id} does not not have an upper limit")
            else:
                upper = limit["upper"]

            if limit["lower"] is None:
                st.text(f"Note: Device {device_id} does not not have an lower limit")
            else:
                lower = limit["lower"]

        if upper is not None:
            fig.add_hline(upper, line={"color": "red"})
            alarm_triggered |= len(df_pred[df_pred.Forecast > upper]) > 0

        if lower is not None:
            fig.add_hline(lower, line={"color": "blue"})
            alarm_triggered |= len(df_pred[df_pred.Forecast < lower]) > 0

    st.plotly_chart(fig)

    if with_alarm:
        if alarm_triggered:
            st.warning("Alarm has been triggered")
        else:
            st.info("Alarm has not been triggered")

    # st.line_chart(df_combined)


d, m = load(device_id)

do_device(d, m, device_id)
