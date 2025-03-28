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
from ctypes import cdll, c_char_p
import json
import os
dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'chronolang.dll'))
chrono = cdll.LoadLibrary(dll_path)
chrono.chrono_parse.argtypes = [c_char_p]
chrono.chrono_parse.restype = c_char_p

datasets = {}

def load_statement(identifier: str, data_source: str):
    
    """Loads data from a CSV file into a dictionary."""

    datasets[identifier] = pd.read_csv(data_source)
    return {"message": f"Loaded {identifier} from {data_source}"}

def export_statement(identifier: str, column: str, destination: str):
    """Exports a dataset column or a forecast result to a CSV file."""
    
    if identifier not in datasets:
        return {"error": f"Dataset '{identifier}' not found."}
    
    data = datasets[identifier]
    
    if column not in data:
        return {"error": f"Column '{column}' not found in dataset '{identifier}'."}

    df = pd.DataFrame({column: data[column]})
    df.to_csv(destination, index=False)
    
    return f"Exported column '{column}' from dataset '{identifier}' to {destination}"


def set_statement(var_name: str, value):

    """Stores a variable in the interpreter's memory."""

    datasets[var_name] = value
    return {"message": f"Variable {var_name} set successfully."}


def select_statement(identifier: str, column: str, op: str = None, date_expr: str = None):

    """Selects data from a dataset based on conditions."""

    if identifier not in datasets:
        return {"error": f"Dataset {identifier} not found."}
    
    df = datasets[identifier]
    if column not in df.columns:
        return {"error": f"Column {column} not found in dataset {identifier}."}
    
    if op and date_expr:
        try:
            if op == "==":
                selected_data = df[df[column] == date_expr]
            elif op == "<":
                selected_data = df[df[column] < date_expr]
            elif op == ">":
                selected_data = df[df[column] > date_expr]
            elif op == ">=":
                selected_data = df[df[column] >= date_expr]
            elif op == "<=":
                selected_data = df[df[column] <= date_expr]
            elif op == "!=":
                selected_data = df[df[column] != date_expr]
            else:
                return {"error": f"Unsupported operator: {op}"}
            
            return {"message": "Selection completed.", "data": selected_data.to_dict()}
        except Exception as e:
            return {"error": f"Selection error: {e}"}
    else:
        return {"message": "No condition applied.", "data": df[column].to_dict()}


def loop_statement(start: int, end: int, body: list):

    """Executes a loop from 'start' to 'end', executing the given statements in each iteration."""

    for i in range(start, end + 1):
        for statement in body:
            interpret({'statements': [statement]})
    return {"message": f"Executed loop from {start} to {end}"}



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


def transform_statement(identifier:str, column: str, time_interval: str):

    """Applies a simple linear trend to forecast the next value."""

    if identifier in datasets:
        df = datasets[identifier]
        x = np.arange(len(df))
        y = df[column]
        coef = np.polyfit(x, y, 1)
        forecast_value = coef[0] * (len(df) + int(time_interval)) + coef[1]
        return {"message": f"Forecasted next {time_interval} value for {column}", "forecast": forecast_value}
    return {"error": f"Data form {identifier} or column {column} not found in the {identifier}."}



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



def interpret(ast: dict):
    
    for statement in ast.get('statements', []):
        stmt_type = statement.get('type')

        if stmt_type == 'Load':
            result = load_statement(statement['id'], statement['path'])
        
        elif stmt_type == 'Set':
            result = set_value(statement['amount'], "int", statement['unit'])  
        
        elif stmt_type == 'Transform':
            result = transform_statement(statement['table'], statement['column'], statement['interval']['amount'])
        elif stmt_type == 'Forecast':
            for key in statement['params'].keys():
                value = statement['params'][key]
                params = {key:value}
            result = forecast_statement(statement['table'], statement['column'], statement['model'], params)
        
        elif stmt_type == 'Stream':
            result = stream_statement(statement['id'], statement['path'])
        
        elif stmt_type == 'Select':
            select_statement(statement['table'],statement['column'],statement['condition']['op'],statement['condition']['date'])
            result = {"message": f"Selection operation on {statement['column']} with condition {statement.get('op', 'None')} {statement.get('dateExpr', 'None')}"}
        
        elif stmt_type == 'Plot':
            #args = {key: value for key, value in statement['args']}
            print(statement['args'].keys())
            args = {}
            for key in statement['args'].keys():
                value =statement['args'][key]
                args[key]=value
            if statement['function'] == 'LINEPLOT':
                result = plot_line(args['data'], args['x_label'], args['y_label'], args.get('title'), args.get('legend'))
            elif statement['function'] == 'histogram':
                result = plot_histogram(args['data'], args['x_label'], args['y_label'], int(args['bins']), args.get('title'))
            elif statement['function'] == 'scatter':
                result = plot_scatter(args['x_data'], args['y_data'], args['x_label'], args['y_label'], args.get('title'))
            elif statement['function'] == 'bar':
                result = plot_bar(args['categories'], args['values'], args['x_label'], args['y_label'], args['orientation'], args.get('title'))
            else:
                result = {"error": f"Unknown plot type: {statement['function']}"}
        
        elif stmt_type == 'Export':
            result = export_statement(statement['table'],statement['column'], statement['to'])
        
        elif stmt_type == 'Loop':
            dictionary = {}
            for _ in range(int(statement['from']),int(statement['to']+1)):
                dictionary['statements'] = statement['body']
                print(dictionary)
                interpret(dictionary)
            
        
        elif stmt_type == 'Clean':
            if statement['action'] == "Remove":
                result = clean_column(statement['table'], statement['column'], "delete")
            elif statement['action'] == "Replace":
                result = clean_column(statement['table'], statement['column'], "replace", statement['replaceWith'])
            else:
                result = {"error": "Unknown clean action"}
        
        else:
            result = {"error": f"Unknown statement type: {stmt_type}"}

        print(result)

f = open("Interpreter\\test.txt", "r")
src = f.read()
json_str = chrono.chrono_parse(src.encode('utf-8'))
ast = json.loads(json_str)
print(ast)
interpret(ast)

#load_statement(1,"Amazon.csv")
#print(datasets[1]["Open"])
#print(forecast_statement(1, "Open", "LSTM", {"steps": 10, "look_back": 6000}))
#x = forecast_statement(1, "Open", "ARIMA", {"steps": 10, "look_back": 6000})

#x=forecast_statement(1, "Open", "Prophet", {"steps": 10, "look_back": 6000})
#print(x)
#export_statement(x,"results")