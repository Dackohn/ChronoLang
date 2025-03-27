import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

datasets = {}

def load_statement(identifier: str, data_source: str):
    
    """Loads data from a CSV file into a dictionary."""

    datasets[identifier] = pd.read_csv(data_source)
    return {"message": f"Loaded {identifier} from {data_source}", "data": datasets[identifier].to_dict()}

def export_statement(result, destination: str):
    """Exports a dataset or a forecast result to a CSV file."""

    if isinstance(result, pd.DataFrame):
        result.to_csv(destination, index=False)
        return f"Exported dataset to {destination}"

    elif isinstance(result, dict):
        if "forecast" in result:
            df = pd.DataFrame(result["forecast"])
            df.to_csv(destination, index=False)
            return f"Exported forecast results to {destination}"
        else:
            return {"error": "No valid data to export."}

    return {"error": "Unsupported data type for export."}


def set_statement():
    """"""

def set_value(var_name: str, var_type: str, value):

    """Sets a variable with the given name, type, and value."""

    if var_type == "int":
        value = int(value)
    elif var_type == "float":
        value = float(value)
    elif var_type == "str":
        value = str(value)
    elif var_type == "array":
        if isinstance(value, list):
            value = [int(v) if isinstance(v, int) else float(v) if isinstance(v, float) else v for v in value]
        else:
            return {"error": f"Invalid value for array: {value}"}
    else:
        return {"error": f"Unsupported type: {var_type}"}
    
    return {var_name: value}

def clean_column(dataset_name: str, column_name: str, action: str, replacement=None):

    """Cleans a specific column in the dataset by deleting null values or replacing them with a given value."""

    if dataset_name not in datasets:
        return {"error": f"Dataset {dataset_name} not found."}
    
    df = datasets[dataset_name]
    
    if column_name not in df.columns:
        return {"error": f"Column {column_name} not found in dataset {dataset_name}."}
    
    if action == "delete":
        df = df.dropna(subset=[column_name])
    elif action == "replace":
        if replacement is None:
            return {"error": "Replacement value must be provided for 'replace' action."}
        df[column_name] = df[column_name].fillna(replacement)
    else:
        return {"error": "Invalid action. Use 'delete' or 'replace'."}
    
    datasets[dataset_name] = df
    return {"message": f"Column {column_name} cleaned successfully in dataset {dataset_name}."}


def transform_statement(column: str, time_interval: str):

    """Applies a simple linear trend to forecast the next value."""

    if column in datasets:
        df = datasets[column]
        x = np.arange(len(df))
        y = df[column]
        coef = np.polyfit(x, y, 1)
        forecast_value = coef[0] * (len(df) + int(time_interval)) + coef[1]
        return {"message": f"Forecasted next {time_interval} value for {column}", "forecast": forecast_value}
    return {"error": f"Column {column} not found."}



def forecast_statement(identifier: str, column: str, model: str, params: dict):
    """Performs forecasting using ARIMA, Prophet, or LSTM."""
    
    if identifier not in datasets:
        return {"error": f"Dataset {identifier} not found."}
    
    df = datasets[identifier]

    if column not in df.columns:
        return {"error": f"Column '{column}' not found in dataset {identifier}."}

    if len(df) < 10:
        return {"error": "Not enough data for forecasting."}

    try:
        if model == "ARIMA":
            order = params.get("order", (1, 1, 1))
            model_fit = ARIMA(df[column], order=order).fit()
            forecast = model_fit.forecast(steps=params.get("steps", 1)).tolist()
            return {"message": f"Forecasted {column} using ARIMA with order {order}", "forecast": forecast}

        elif model == "Prophet":
            if "Date" not in df.columns:
                return {"error": "Prophet requires a 'Date' column."}
            
            prophet_df = df.rename(columns={"Date": "ds", column: "y"})[["ds", "y"]]
            prophet = Prophet()
            prophet.fit(prophet_df)
            
            future = prophet.make_future_dataframe(periods=params.get("steps", 1))
            forecast = prophet.predict(future)[["ds", "yhat"]].tail(params.get("steps", 1)).to_dict(orient="records")
            return {"message": f"Forecasted {column} using Prophet.", "forecast": forecast}

        elif model == "LSTM":
            steps = params.get("steps", 1)
            look_back = params.get("look_back", 5)

            data = df[column].values.reshape(-1, 1)
            x_train, y_train = [], []

            for i in range(len(data) - look_back):
                x_train.append(data[i:i+look_back])
                y_train.append(data[i+look_back])

            x_train, y_train = np.array(x_train), np.array(y_train)
            
            model = Sequential([
                LSTM(50, activation="relu", return_sequences=True, input_shape=(look_back, 1)),
                LSTM(50, activation="relu"),
                Dense(1)
            ])
            model.compile(optimizer="adam", loss="mse")
            model.fit(x_train, y_train, epochs=50, verbose=0)

            x_input = data[-look_back:].reshape(1, look_back, 1)
            forecast = [model.predict(x_input, verbose=0)[0][0] for _ in range(steps)]
            
            return {"message": f"Forecasted {column} using LSTM.", "forecast": forecast}

        else:
            return {"error": f"Unsupported model type: {model}"}

    except Exception as e:
        return {"error": f"Forecasting error: {e}"}


def stream_statement(identifier: str, stream_source: str, chunk_size: int = 100):

    """Streams data in chunks from a CSV file."""

    if identifier not in datasets:
        datasets[identifier] = pd.DataFrame()
    try:
        chunk = pd.read_csv(stream_source, chunksize=chunk_size)
        for new_data in chunk:
            datasets[identifier] = pd.concat([datasets[identifier], new_data], ignore_index=True)
            return {"message": f"Streamed {len(new_data)} rows into {identifier}", "data": datasets[identifier].to_dict()}
    except Exception as e:
        return {"error": f"Streaming error: {e}"}

def generate_plot():
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return encoded_image

def plot_line(data, x_label: str, y_label: str, title=None, legend=None):

    """Generates a line plot."""

    plt.figure()
    for series in data:
        plt.plot(series)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)
    if legend:
        plt.legend(legend)
    encoded_image = generate_plot()
    return {"message": "Line plot generated.", "plot": encoded_image}

def plot_histogram(data, x_label: str, y_label: str, bins: int, title=None):

    """Generates a histogram."""

    plt.figure()
    plt.hist(data, bins=bins)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)
    encoded_image = generate_plot()
    return {"message": "Histogram generated.", "plot": encoded_image}

def plot_scatter(x_data, y_data, x_label: str, y_label: str, title=None):

    """Generates a scatter plot."""

    plt.figure()
    plt.scatter(x_data, y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)
    encoded_image = generate_plot()
    return {"message": "Scatter plot generated.", "plot": encoded_image}

def plot_bar(categories, values, x_label: str, y_label: str, orientation: str, title=None):

    """Generates a bar plot."""

    plt.figure()
    if orientation == "vertical":
        plt.bar(categories, values)
    else:
        plt.barh(categories, values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)
    encoded_image = generate_plot()
    return {"message": "Bar plot generated.", "plot": encoded_image}


load_statement(1,"Amazon.csv")
print(datasets[1]["Open"])
#print(forecast_statement(1, "Open", "LSTM", {"steps": 10, "look_back": 6000}))
#x = forecast_statement(1, "Open", "ARIMA", {"steps": 10, "look_back": 6000})

x=forecast_statement(1, "Open", "Prophet", {"steps": 10, "look_back": 6000})
print(x)
export_statement(x,"results")