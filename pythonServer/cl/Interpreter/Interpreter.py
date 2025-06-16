from datetime import datetime

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
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
from ast import literal_eval
import traceback
from sklearn.preprocessing import MinMaxScaler

# Load the ChronoLang DLL
dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'chronolang.dll'))
chrono = cdll.LoadLibrary(dll_path)
chrono.chrono_parse.argtypes = [c_char_p]
chrono.chrono_parse.restype = c_char_p

datasets = {}
variables = {}


def load_statement(identifier: str, data_source: str):
    """Loads data from a CSV file into a dictionary."""
    try:
        datasets[identifier] = pd.read_csv(data_source)
        return {"status": "success", "message": f"Loaded {identifier} from {data_source}"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to load {identifier}: {str(e)}"}


def export_statement(identifier: str, column: str, destination: str):
    """Exports a dataset column or a forecast result to a CSV file."""

    # Handle variable references (starting with $)
    if identifier.startswith('$'):
        var_name = identifier[1:]  # Remove $ prefix
        if var_name not in variables:
            return {"status": "error", "message": f"Variable ${var_name} not found."}

        # Get the data from the variable
        if isinstance(variables[var_name], dict):
            if 'data' in variables[var_name]:
                # It's a select result
                var_data = variables[var_name]['data']
                df = pd.DataFrame({column: list(var_data.values())})
            elif 'forecast' in variables[var_name]:
                # It's a forecast result
                df = pd.DataFrame({column: variables[var_name]['forecast']})
            else:
                return {"status": "error", "message": f"Variable ${var_name} doesn't contain exportable data."}
        elif hasattr(variables[var_name], 'columns'):
            df = variables[var_name]
        else:
            return {"status": "error", "message": f"Variable ${var_name} is not a valid dataset."}
    else:
        # Handle regular dataset references
        if identifier not in datasets:
            return {"status": "error", "message": f"Dataset '{identifier}' not found."}
        df = datasets[identifier]

    # For variables, we might not have the column in the dataframe
    if column not in df.columns:
        # If it's a single-column dataframe, use the first column
        if len(df.columns) == 1:
            column = df.columns[0]
        else:
            return {"status": "error", "message": f"Column '{column}' not found."}

    try:
        export_df = pd.DataFrame({column: df[column]})
        export_df.to_csv(destination, index=False)
        return {"status": "success", "message": f"Exported column '{column}' to {destination}"}
    except Exception as e:
        return {"status": "error", "message": f"Export failed: {str(e)}"}


def set_statement(var_name: str, value):
    """Stores a variable in the interpreter's memory."""
    try:
        datasets[var_name] = value
        return {"status": "success", "message": f"Variable {var_name} set successfully."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to set variable: {str(e)}"}


def select_statement(identifier: str, column: str, op: str = None, date_expr: str = None, alias: str = None):
    # Handle variable references (starting with $)
    if identifier.startswith('$'):
        var_name = identifier[1:]  # Remove $ prefix
        if var_name not in variables:
            return {"status": "error", "message": f"Variable ${var_name} not found."}

        # Get the data from the variable
        if isinstance(variables[var_name], dict) and 'data' in variables[var_name]:
            # If it's a result from a previous select, extract the data
            var_data = variables[var_name]['data']
            df = pd.DataFrame({column: list(var_data.values())})
        elif hasattr(variables[var_name], 'columns'):
            # If it's a DataFrame
            df = variables[var_name]
        else:
            return {"status": "error", "message": f"Variable ${var_name} is not a valid dataset."}
    else:
        # Handle regular dataset references
        if identifier not in datasets:
            return {"status": "error", "message": f"Dataset {identifier} not found."}
        df = datasets[identifier]

    if column not in df.columns:
        return {"status": "error", "message": f"Column {column} not found in dataset {identifier}."}

    if op and date_expr:
        try:
            comparison_column = 'Date'

            if comparison_column not in df.columns:
                date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                if date_columns:
                    comparison_column = date_columns[0]
                else:
                    return {"status": "error",
                            "message": f"No date column found in dataset {identifier} for comparison."}

            col_dtype = df[comparison_column].dtype

            if 'datetime' in str(col_dtype) or 'date' in str(col_dtype):
                date_expr_converted = pd.to_datetime(date_expr)
            elif 'int' in str(col_dtype) or 'float' in str(col_dtype):
                date_expr_converted = pd.to_datetime(date_expr).timestamp()
                if df[comparison_column].max() < 20000101 and df[comparison_column].min() > 19000101:
                    date_obj = datetime.strptime(date_expr, "%Y-%m-%d")
                    date_expr_converted = int(date_obj.strftime("%Y%m%d"))
            else:
                date_expr_converted = date_expr

            # Perform the comparison
            if op == "==":
                selected_data = df[df[comparison_column] == date_expr_converted]
            elif op == "<":
                selected_data = df[df[comparison_column] < date_expr_converted]
            elif op == ">":
                selected_data = df[df[comparison_column] > date_expr_converted]
            elif op == ">=":
                selected_data = df[df[comparison_column] >= date_expr_converted]
            elif op == "<=":
                selected_data = df[df[comparison_column] <= date_expr_converted]
            elif op == "!=":
                selected_data = df[df[comparison_column] != date_expr_converted]
            else:
                return {"status": "error", "message": f"Unsupported operator: {op}"}

            # Extract the requested column from the filtered data
            result_data = selected_data[column].to_dict() if not selected_data.empty else {}

            result = {
                "status": "success",
                "message": f"Selected {len(result_data)} rows from {column} where {comparison_column} {op} {date_expr}",
                "data": result_data
            }

        except Exception as e:
            result = {"status": "error", "message": f"Selection error: {str(e)}"}
    else:
        # No condition provided, return the entire column
        result = {"status": "success", "message": "No condition applied.", "data": df[column].to_dict()}

    # Store the result in variables if alias is provided
    if alias:
        variables[alias] = result
        result["message"] += f" (saved as ${alias})"

    return result


def loop_statement(start: int, end: int, body: list):
    """Executes a loop from 'start' to 'end', executing the given statements in each iteration."""
    try:
        results = []
        for i in range(start, end + 1):
            for statement in body:
                result = interpret({'statements': [statement]})
                results.append(result)
        return {"status": "success", "message": f"Executed loop from {start} to {end}", "results": results}
    except Exception as e:
        return {"status": "error", "message": f"Loop execution failed: {str(e)}"}


def set_value(var_name: str, var_type: str, value):
    """Sets a variable with the given name, type, and value."""
    try:
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
                return {"status": "error", "message": f"Invalid value for array: {value}"}
        else:
            return {"status": "error", "message": f"Unsupported type: {var_type}"}

        return {"status": "success", var_name: value}
    except Exception as e:
        return {"status": "error", "message": f"Failed to set value: {str(e)}"}


def clean_column(dataset_name: str, column_name: str, action: str, replacement=None):
    """Cleans a specific column in the dataset by deleting null values or replacing them with a given value."""
    if dataset_name not in datasets:
        return {"status": "error", "message": f"Dataset {dataset_name} not found."}

    df = datasets[dataset_name]

    if column_name not in df.columns:
        return {"status": "error", "message": f"Column {column_name} not found in dataset {dataset_name}."}

    try:
        if action == "delete":
            df = df.dropna(subset=[column_name])
        elif action == "replace":
            if replacement is None:
                return {"status": "error", "message": "Replacement value must be provided for 'replace' action."}
            df[column_name] = df[column_name].fillna(replacement)
        else:
            return {"status": "error", "message": "Invalid action. Use 'delete' or 'replace'."}

        datasets[dataset_name] = df
        return {"status": "success", "message": f"Column {column_name} cleaned successfully in dataset {dataset_name}."}
    except Exception as e:
        return {"status": "error", "message": f"Cleaning failed: {str(e)}"}


def transform_statement(identifier: str, column: str, time_interval: str, alias: str = None):
    """Applies a simple linear trend to forecast the next value."""
    try:
        # Handle variable references (starting with $)
        if identifier.startswith('$'):
            var_name = identifier[1:]  # Remove $ prefix
            if var_name not in variables:
                return {"status": "error", "message": f"Variable ${var_name} not found."}

            # Get the data from the variable
            if isinstance(variables[var_name], dict) and 'data' in variables[var_name]:
                var_data = variables[var_name]['data']
                df = pd.DataFrame({column: list(var_data.values())})
            elif hasattr(variables[var_name], 'columns'):
                df = variables[var_name]
            else:
                return {"status": "error", "message": f"Variable ${var_name} is not a valid dataset."}
        else:
            # Handle regular dataset references
            if identifier not in datasets:
                return {"status": "error", "message": f"Dataset {identifier} not found."}
            df = datasets[identifier]

        if column not in df.columns:
            return {"status": "error", "message": f"Column {column} not found."}

        x = np.arange(len(df))
        y = df[column]
        coef = np.polyfit(x, y, 1)
        forecast_value = coef[0] * (len(df) + int(time_interval)) + coef[1]

        result = {
            "status": "success",
            "message": f"Forecasted next {time_interval} value for {column}",
            "forecast": float(forecast_value)
        }

        # Store the result in variables if alias is provided
        if alias:
            variables[alias] = result
            result["message"] += f" (saved as ${alias})"

        return result

    except Exception as e:
        return {"status": "error", "message": f"Transform failed: {str(e)}"}


# Add this global dictionary to store variables created with AS
variables = {}


def select_statement(identifier: str, column: str, op: str = None, date_expr: str = None, alias: str = None):
    # Handle variable references (starting with $)
    if identifier.startswith('$'):
        var_name = identifier[1:]  # Remove $ prefix
        if var_name not in variables:
            return {"status": "error", "message": f"Variable ${var_name} not found."}

        # Get the data from the variable
        if isinstance(variables[var_name], dict) and 'data' in variables[var_name]:
            # If it's a result from a previous select, extract the data
            var_data = variables[var_name]['data']
            df = pd.DataFrame({column: list(var_data.values())})
        elif hasattr(variables[var_name], 'columns'):
            # If it's a DataFrame
            df = variables[var_name]
        else:
            return {"status": "error", "message": f"Variable ${var_name} is not a valid dataset."}
    else:
        # Handle regular dataset references
        if identifier not in datasets:
            return {"status": "error", "message": f"Dataset {identifier} not found."}
        df = datasets[identifier]

    if column not in df.columns:
        return {"status": "error", "message": f"Column {column} not found in dataset {identifier}."}

    if op and date_expr:
        try:
            comparison_column = 'Date'

            if comparison_column not in df.columns:
                date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                if date_columns:
                    comparison_column = date_columns[0]
                else:
                    return {"status": "error",
                            "message": f"No date column found in dataset {identifier} for comparison."}

            col_dtype = df[comparison_column].dtype

            if 'datetime' in str(col_dtype) or 'date' in str(col_dtype):
                import pandas as pd
                date_expr_converted = pd.to_datetime(date_expr)
            elif 'int' in str(col_dtype) or 'float' in str(col_dtype):
                import pandas as pd
                import time
                date_expr_converted = pd.to_datetime(date_expr).timestamp()
                if df[comparison_column].max() < 20000101 and df[comparison_column].min() > 19000101:
                    from datetime import datetime
                    date_obj = datetime.strptime(date_expr, "%Y-%m-%d")
                    date_expr_converted = int(date_obj.strftime("%Y%m%d"))
            else:
                date_expr_converted = date_expr

            # Perform the comparison
            if op == "==":
                selected_data = df[df[comparison_column] == date_expr_converted]
            elif op == "<":
                selected_data = df[df[comparison_column] < date_expr_converted]
            elif op == ">":
                selected_data = df[df[comparison_column] > date_expr_converted]
            elif op == ">=":
                selected_data = df[df[comparison_column] >= date_expr_converted]
            elif op == "<=":
                selected_data = df[df[comparison_column] <= date_expr_converted]
            elif op == "!=":
                selected_data = df[df[comparison_column] != date_expr_converted]
            else:
                return {"status": "error", "message": f"Unsupported operator: {op}"}

            # Extract the requested column from the filtered data
            result_data = selected_data[column].to_dict() if not selected_data.empty else {}

            result = {
                "status": "success",
                "message": f"Selected {len(result_data)} rows from {column} where {comparison_column} {op} {date_expr}",
                "data": result_data
            }

        except Exception as e:
            result = {"status": "error", "message": f"Selection error: {str(e)}"}
    else:
        # No condition provided, return the entire column
        result = {"status": "success", "message": "No condition applied.", "data": df[column].to_dict()}

    # Store the result in variables if alias is provided
    if alias:
        variables[alias] = result
        result["message"] += f" (saved as ${alias})"

    return result


def forecast_statement(identifier: str, column: str, model: str, params: dict, alias: str = None):
    """Performs forecasting using ARIMA, Prophet, or LSTM."""

    # Handle variable references (starting with $)
    if identifier.startswith('$'):
        var_name = identifier[1:]  # Remove $ prefix
        if var_name not in variables:
            return {"status": "error", "message": f"Variable ${var_name} not found."}

        # Get the data from the variable
        if isinstance(variables[var_name], dict):
            if 'data' in variables[var_name]:
                # Create a DataFrame from the variable data with proper indexing
                var_data = variables[var_name]['data']
                if isinstance(var_data, dict):
                    # Convert dict to list of values, maintaining order
                    data_values = list(var_data.values())
                    df = pd.DataFrame({column: data_values})
                    # Create a proper date index if needed for Prophet
                    if model == "Prophet":
                        df['Date'] = pd.date_range(start='2000-01-01', periods=len(data_values), freq='D')
                else:
                    df = pd.DataFrame({column: var_data})
                    if model == "Prophet":
                        df['Date'] = pd.date_range(start='2000-01-01', periods=len(var_data), freq='D')
            elif 'forecast' in variables[var_name]:
                # If it's a previous forecast result, use the forecast data
                forecast_data = variables[var_name]['forecast']
                if isinstance(forecast_data, list):
                    df = pd.DataFrame({column: forecast_data})
                    if model == "Prophet":
                        df['Date'] = pd.date_range(start='2000-01-01', periods=len(forecast_data), freq='D')
                else:
                    return {"status": "error", "message": f"Invalid forecast data in variable ${var_name}."}
            else:
                return {"status": "error",
                        "message": f"Variable ${var_name} doesn't contain valid data for forecasting."}
        elif hasattr(variables[var_name], 'columns'):
            # If it's already a DataFrame
            df = variables[var_name]
        else:
            return {"status": "error", "message": f"Variable ${var_name} is not a valid dataset for forecasting."}
    else:
        # Handle regular dataset references
        if identifier not in datasets:
            return {"status": "error", "message": f"Dataset {identifier} not found."}
        df = datasets[identifier]

    # Check if the column exists, if not and there's only one column, use it
    if column not in df.columns:
        if len(df.columns) == 1:
            column = df.columns[0]
        else:
            return {"status": "error",
                    "message": f"Column '{column}' not found in dataset {identifier}. Available columns: {list(df.columns)}"}

    if len(df) < 10:
        return {"status": "error", "message": "Not enough data for forecasting (minimum 10 data points required)."}

    try:
        if model == "ARIMA":
            order = params.get("order", (1, 1, 1))
            # Handle model_order parameter as well for backward compatibility
            if "model_order" in params:
                order = (params["model_order"], 1, 1)

            model_fit = ARIMA(df[column], order=order).fit()
            forecast = model_fit.forecast(steps=params.get("steps", 1)).tolist()
            result = {
                "status": "success",
                "message": f"Forecasted {column} using ARIMA with order {order}",
                "forecast": forecast
            }

        elif model == "Prophet":
            # Check if Date column exists, if not create one
            if "Date" not in df.columns:
                df = df.copy()  # Don't modify original
                df['Date'] = pd.date_range(start='2000-01-01', periods=len(df), freq='D')

            # Prepare data for Prophet
            prophet_df = df[['Date', column]].copy()
            prophet_df.columns = ['ds', 'y']

            # Remove any NaN values
            prophet_df = prophet_df.dropna()

            if len(prophet_df) < 2:
                return {"status": "error", "message": "Not enough valid data points for Prophet forecasting."}

            prophet = Prophet()
            prophet.fit(prophet_df)

            future = prophet.make_future_dataframe(periods=params.get("steps", 1))
            forecast_result = prophet.predict(future)

            # Get only the forecasted values (not the fitted values)
            forecast = forecast_result['yhat'].tail(params.get("steps", 1)).tolist()

            result = {
                "status": "success",
                "message": f"Forecasted {column} using Prophet for {params.get('steps', 1)} periods.",
                "forecast": forecast
            }

        elif model == "LSTM":
            steps = params.get("steps", 1)
            look_back = params.get("look_back", 5)
            layers = params.get("layers", 3)
            epochs = params.get("epochs", 50)

            # Get numeric data and handle NaN values
            data = df[column].dropna().values.astype(float).reshape(-1, 1)

            if len(data) < look_back + 1:
                return {"status": "error",
                        "message": f"Not enough data for LSTM (need at least {look_back + 1} points)."}

            # Normalize data

            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data)

            x_train, y_train = [], []
            for i in range(len(data_scaled) - look_back):
                x_train.append(data_scaled[i:i + look_back])
                y_train.append(data_scaled[i + look_back])

            x_train, y_train = np.array(x_train), np.array(y_train)

            # Build model with dynamic number of layers
            model = Sequential()

            # First LSTM layer (always present)
            if layers > 1:
                model.add(LSTM(50, activation="relu", return_sequences=True, input_shape=(look_back, 1)))

                # Add intermediate LSTM layers
                for i in range(1, layers - 1):
                    model.add(LSTM(50, activation="relu", return_sequences=True))

                # Last LSTM layer (no return_sequences)
                model.add(LSTM(50, activation="relu"))
            else:
                # Single LSTM layer
                model.add(LSTM(50, activation="relu", input_shape=(look_back, 1)))

            # Output layer
            model.add(Dense(1))

            model.compile(optimizer="adam", loss="mse")
            model.fit(x_train, y_train, epochs=epochs, verbose=0)

            # Generate forecasts
            forecast = []
            current_batch = data_scaled[-look_back:].reshape(1, look_back, 1)

            for _ in range(steps):
                current_pred = model.predict(current_batch, verbose=0)[0]
                forecast.append(current_pred[0])
                # Update batch for next prediction
                current_batch = np.append(current_batch[:, 1:, :], current_pred.reshape(1, 1, 1), axis=1)

            # Inverse transform to get actual values
            forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten().tolist()

            result = {
                "status": "success",
                "message": f"Forecasted {column} using LSTM with {layers} layers for {steps} steps.",
                "forecast": forecast
            }
        else:
            return {"status": "error", "message": f"Unsupported model type: {model}"}

        # Store the result in variables if alias is provided
        if alias:
            variables[alias] = result
            result["message"] += f" (saved as ${alias})"

        return result

    except Exception as e:
        import traceback
        return {"status": "error", "message": f"Forecasting error: {str(e)}", "traceback": traceback.format_exc()}


def stream_statement(identifier: str, stream_source: str, chunk_size: int = 100):
    """Streams data in chunks from a CSV file."""
    if identifier not in datasets:
        datasets[identifier] = pd.DataFrame()
    try:
        chunk = pd.read_csv(stream_source, chunksize=chunk_size)
        for new_data in chunk:
            datasets[identifier] = pd.concat([datasets[identifier], new_data], ignore_index=True)
            return {
                "status": "success",
                "message": f"Streamed {len(new_data)} rows into {identifier}",
                "data": datasets[identifier].to_dict()
            }
    except Exception as e:
        return {"status": "error", "message": f"Streaming error: {str(e)}"}


def generate_plot():
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close()  # Close the plot to free memory
    return encoded_image


def safe_parse(value):
    if isinstance(value, str):
        try:
            import ast
            parsed = ast.literal_eval(value)
            return parsed
        except Exception:
            return value
    return value


def plot_line(data, x_label: str, y_label: str, title=None, legend=None):
    plt.figure(figsize=(10, 6))

    if data is None or (isinstance(data, list) and len(data) == 0):
        return {"status": "error", "message": "No data provided for plotting"}

    try:
        if (isinstance(data, list) and all(isinstance(x, (int, float, np.number)) for x in data)) or \
                isinstance(data, np.ndarray) and data.ndim == 1:
            plt.plot(data, label='Series')
            use_legend = True

        elif isinstance(data, list) and all(isinstance(series, (list, np.ndarray)) for series in data):
            use_legend = False
            for i, series in enumerate(data):
                if len(series) > 0:
                    plt.plot(series, label=f'Series {i + 1}')
                    use_legend = True

        elif hasattr(data, 'plot'):
            data.plot(ax=plt.gca())
            use_legend = True

        else:
            try:
                data_array = np.array(data)
                if data_array.ndim == 1:
                    plt.plot(data_array, label='Series')
                    use_legend = True
                elif data_array.ndim == 2:
                    for i in range(
                            data_array.shape[1] if data_array.shape[0] < data_array.shape[1] else data_array.shape[0]):
                        plt.plot(data_array[i] if data_array.shape[0] > data_array.shape[1] else data_array[:, i],
                                 label=f'Series {i + 1}')
                    use_legend = True
                else:
                    return {"status": "error", "message": "Data has too many dimensions for plotting"}
            except Exception as e:
                return {"status": "error", "message": f"Failed to convert data for plotting: {str(e)}"}

    except Exception as e:
        return {"status": "error", "message": f"Error during plot generation: {str(e)}"}

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)

    if legend:
        if isinstance(legend, str):
            try:
                import ast
                parsed_legend = ast.literal_eval(legend)
                legend = parsed_legend if isinstance(parsed_legend, list) else [legend]
            except Exception:
                legend = [legend]
        plt.legend(legend)
    elif use_legend:
        plt.legend()

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    # plt.show()

    encoded_image = generate_plot()
    return {"status": "success", "message": "Line plot generated successfully.", "plot": encoded_image}


def plot_histogram(data, x_label: str, y_label: str, bins: int, title=None):
    """Generates a histogram with support for dataset references."""
    try:
        # Handle dataset references (e.g., sales_data.High)
        if isinstance(data, str) and '.' in data and data.count('.') == 1:
            dataset_name, column_name = data.split('.')
            if dataset_name in datasets and column_name in datasets[dataset_name].columns:
                data = datasets[dataset_name][column_name].tolist()
            else:
                return {"status": "error", "message": f"Dataset reference '{data}' not found"}

        # Handle variable references (e.g., $var_1)
        elif isinstance(data, str) and data.startswith('$'):
            var_name = data[1:]
            if var_name in variables:
                if isinstance(variables[var_name], dict) and 'data' in variables[var_name]:
                    data = list(variables[var_name]['data'].values())
                elif isinstance(variables[var_name], dict) and 'forecast' in variables[var_name]:
                    data = variables[var_name]['forecast']
                else:
                    return {"status": "error", "message": f"Variable '{data}' doesn't contain plottable data"}
            else:
                return {"status": "error", "message": f"Variable '{data}' not found"}

        # Handle list of dataset references
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], str) and '.' in str(data[0]):
            combined_data = []
            for item in data:
                if isinstance(item, str) and '.' in item and item.count('.') == 1:
                    dataset_name, column_name = item.split('.')
                    if dataset_name in datasets and column_name in datasets[dataset_name].columns:
                        combined_data.extend(datasets[dataset_name][column_name].tolist())
                    else:
                        return {"status": "error", "message": f"Dataset reference '{item}' not found"}
                else:
                    combined_data.extend(item if isinstance(item, list) else [item])
            data = combined_data

        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=bins)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if title:
            plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        encoded_image = generate_plot()
        return {"status": "success", "message": "Histogram generated.", "plot": encoded_image}
    except Exception as e:
        return {"status": "error", "message": f"Histogram generation failed: {str(e)}"}


def plot_scatter(x_data, y_data, x_label: str, y_label: str, title=None):
    """Generates a scatter plot with support for dataset references."""
    try:
        # Process x_data
        if isinstance(x_data, str) and '.' in x_data and x_data.count('.') == 1:
            dataset_name, column_name = x_data.split('.')
            if dataset_name in datasets and column_name in datasets[dataset_name].columns:
                x_data = datasets[dataset_name][column_name].tolist()
            else:
                return {"status": "error", "message": f"Dataset reference '{x_data}' not found"}
        elif isinstance(x_data, str) and x_data.startswith('$'):
            var_name = x_data[1:]
            if var_name in variables:
                if isinstance(variables[var_name], dict) and 'data' in variables[var_name]:
                    x_data = list(variables[var_name]['data'].values())
                elif isinstance(variables[var_name], dict) and 'forecast' in variables[var_name]:
                    x_data = variables[var_name]['forecast']
                else:
                    return {"status": "error", "message": f"Variable '{x_data}' doesn't contain plottable data"}
            else:
                return {"status": "error", "message": f"Variable '{x_data}' not found"}

        # Process y_data
        if isinstance(y_data, str) and '.' in y_data and y_data.count('.') == 1:
            dataset_name, column_name = y_data.split('.')
            if dataset_name in datasets and column_name in datasets[dataset_name].columns:
                y_data = datasets[dataset_name][column_name].tolist()
            else:
                return {"status": "error", "message": f"Dataset reference '{y_data}' not found"}
        elif isinstance(y_data, str) and y_data.startswith('$'):
            var_name = y_data[1:]
            if var_name in variables:
                if isinstance(variables[var_name], dict) and 'data' in variables[var_name]:
                    y_data = list(variables[var_name]['data'].values())
                elif isinstance(variables[var_name], dict) and 'forecast' in variables[var_name]:
                    y_data = variables[var_name]['forecast']
                else:
                    return {"status": "error", "message": f"Variable '{y_data}' doesn't contain plottable data"}
            else:
                return {"status": "error", "message": f"Variable '{y_data}' not found"}

        # Ensure both datasets have the same length
        if len(x_data) != len(y_data):
            min_len = min(len(x_data), len(y_data))
            x_data = x_data[:min_len]
            y_data = y_data[:min_len]

        plt.figure(figsize=(10, 6))
        plt.scatter(x_data, y_data, alpha=0.6)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if title:
            plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        encoded_image = generate_plot()
        return {"status": "success", "message": "Scatter plot generated.", "plot": encoded_image}
    except Exception as e:
        return {"status": "error", "message": f"Scatter plot generation failed: {str(e)}"}


def plot_bar(categories, values, x_label: str, y_label: str, orientation: str, title=None):
    """Generates a bar plot with support for dataset references."""
    try:
        # Process categories
        if isinstance(categories, str) and '.' in categories and categories.count('.') == 1:
            dataset_name, column_name = categories.split('.')
            if dataset_name in datasets and column_name in datasets[dataset_name].columns:
                categories = datasets[dataset_name][column_name].tolist()
            else:
                return {"status": "error", "message": f"Dataset reference '{categories}' not found"}
        elif isinstance(categories, str) and categories.startswith('$'):
            var_name = categories[1:]
            if var_name in variables:
                if isinstance(variables[var_name], dict) and 'data' in variables[var_name]:
                    categories = list(variables[var_name]['data'].values())
                elif isinstance(variables[var_name], dict) and 'forecast' in variables[var_name]:
                    categories = variables[var_name]['forecast']
                else:
                    return {"status": "error", "message": f"Variable '{categories}' doesn't contain plottable data"}
            else:
                return {"status": "error", "message": f"Variable '{categories}' not found"}

        # Process values
        if isinstance(values, str) and '.' in values and values.count('.') == 1:
            dataset_name, column_name = values.split('.')
            if dataset_name in datasets and column_name in datasets[dataset_name].columns:
                values = datasets[dataset_name][column_name].tolist()
            else:
                return {"status": "error", "message": f"Dataset reference '{values}' not found"}
        elif isinstance(values, str) and values.startswith('$'):
            var_name = values[1:]
            if var_name in variables:
                if isinstance(variables[var_name], dict) and 'data' in variables[var_name]:
                    values = list(variables[var_name]['data'].values())
                elif isinstance(variables[var_name], dict) and 'forecast' in variables[var_name]:
                    values = variables[var_name]['forecast']
                else:
                    return {"status": "error", "message": f"Variable '{values}' doesn't contain plottable data"}
            else:
                return {"status": "error", "message": f"Variable '{values}' not found"}

        # Handle case where categories might be numeric indices and we need to create labels
        if isinstance(categories[0], (int, float)) and not isinstance(categories, range):
            # If categories are numeric, create string labels
            categories = [f"Item {i}" for i in range(len(values))]

        # Ensure both datasets have the same length
        if len(categories) != len(values):
            min_len = min(len(categories), len(values))
            categories = categories[:min_len]
            values = values[:min_len]

        plt.figure(figsize=(10, 6))
        if orientation.lower() == "vertical":
            plt.bar(categories, values)
            plt.xticks(rotation=45 if len(str(categories[0])) > 5 else 0)
        else:
            plt.barh(categories, values)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if title:
            plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        encoded_image = generate_plot()
        return {"status": "success", "message": "Bar plot generated.", "plot": encoded_image}
    except Exception as e:
        return {"status": "error", "message": f"Bar plot generation failed: {str(e)}"}


def interpret(ast: dict):
    """Interprets the AST and returns JSON results."""
    results = []

    for statement in ast.get('statements', []):
        stmt_type = statement.get('type')

        try:
            if stmt_type == 'Load':
                result = load_statement(statement['id'], statement['path'])

            elif stmt_type == 'Set':
                result = set_value(statement['amount'], "int", statement['unit'])

            elif stmt_type == 'Transform':
                alias = statement.get('alias')
                result = transform_statement(statement['table'], statement['column'], statement['interval']['amount'],
                                             alias)

            elif stmt_type == 'Forecast':
                params = {}
                for key, value in statement['params'].items():
                    params[key] = value

                # Handle both 'table' and 'variable' fields
                identifier = statement.get('table') or statement.get('variable')
                if identifier and not identifier.startswith('$'):
                    identifier = '$' + identifier  # Add $ prefix for variables

                alias = statement.get('alias')
                result = forecast_statement(identifier, statement.get('column', 'value'), statement['model'], params,
                                            alias)

            elif stmt_type == 'Stream':
                result = stream_statement(statement['id'], statement['path'])

            elif stmt_type == 'Select':
                column_to_retrieve = statement['column']
                alias = statement.get('alias')

                if 'condition' in statement and 'op' in statement['condition'] and 'date' in statement['condition']:
                    op = statement['condition']['op']
                    date_value = statement['condition']['date']
                    result = select_statement(statement['table'], column_to_retrieve, op, date_value, alias)
                else:
                    result = select_statement(statement['table'], column_to_retrieve, alias=alias)

                # Keep the old behavior for backward compatibility
                if 'data' in result and not alias:
                    import pandas as pd
                    selected_data = pd.DataFrame({column_to_retrieve: result['data'].values()})
                    datasets[f"{statement['table']}_selected"] = selected_data

            elif stmt_type == 'Plot':
                args = {}
                for key, value in statement['args'].items():
                    # Handle variable references in plot data
                    if isinstance(value, str) and value.startswith('$'):
                        var_name = value[1:]
                        if var_name in variables:
                            if isinstance(variables[var_name], dict) and 'data' in variables[var_name]:
                                args[key] = list(variables[var_name]['data'].values())
                            elif isinstance(variables[var_name], dict) and 'forecast' in variables[var_name]:
                                args[key] = variables[var_name]['forecast']
                            else:
                                args[key] = value
                        else:
                            args[key] = value
                        continue

                    # Handle comma-separated dataset references
                    if isinstance(value, str) and ',' in value and '.' in value:
                        parts = [part.strip() for part in value.split(',')]
                        all_parts_are_refs = all('.' in part and part.count('.') == 1 for part in parts)

                        if all_parts_are_refs:
                            try:
                                processed_list = []
                                for part in parts:
                                    dataset_name, column_name = part.split('.')
                                    if dataset_name in datasets and column_name in datasets[dataset_name].columns:
                                        processed_list.append(datasets[dataset_name][column_name].tolist())
                                    else:
                                        raise ValueError(f"Invalid dataset reference: {part}")

                                args[key] = processed_list
                                continue
                            except Exception as e:
                                pass

                    # Handle bracketed lists with dataset references
                    if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                        inner_content = value[1:-1].strip()

                        if ',' in inner_content and all('.' in part.strip() for part in inner_content.split(',')):
                            try:
                                parts = [part.strip() for part in inner_content.split(',')]
                                processed_list = []

                                for part in parts:
                                    dataset_name, column_name = part.split('.')
                                    if dataset_name in datasets and column_name in datasets[dataset_name].columns:
                                        processed_list.append(datasets[dataset_name][column_name].tolist())
                                    else:
                                        raise ValueError(f"Invalid dataset reference: {part}")

                                args[key] = processed_list
                                continue
                            except Exception as e:
                                pass

                        if '.' in inner_content and inner_content.count('.') == 1:
                            try:
                                dataset_name, column_name = inner_content.split('.')
                                if dataset_name in datasets and column_name in datasets[dataset_name].columns:
                                    args[key] = [datasets[dataset_name][column_name].tolist()]
                                    continue
                            except Exception as e:
                                pass

                    # Handle single dataset references
                    if isinstance(value, str) and '.' in value and value.count('.') == 1:
                        try:
                            dataset_name, column_name = value.split('.')
                            if dataset_name in datasets and column_name in datasets[dataset_name].columns:
                                args[key] = datasets[dataset_name][column_name].tolist()
                                continue
                        except Exception as e:
                            pass

                    # Handle string literals and lists
                    if isinstance(value, str):
                        try:
                            import ast as python_ast
                            parsed_value = python_ast.literal_eval(value)

                            if isinstance(parsed_value, list):
                                processed_list = []
                                all_elements_processed = True

                                for item in parsed_value:
                                    if isinstance(item, str) and '.' in item and item.count('.') == 1:
                                        dataset_name, column_name = item.split('.')
                                        if dataset_name in datasets and column_name in datasets[dataset_name].columns:
                                            processed_list.append(datasets[dataset_name][column_name].tolist())
                                        else:
                                            processed_list.append(item)
                                            all_elements_processed = False
                                    else:
                                        processed_list.append(item)
                                        all_elements_processed = False

                                if all_elements_processed or len(processed_list) != len(parsed_value):
                                    args[key] = processed_list
                                    continue
                                else:
                                    args[key] = parsed_value
                                    continue
                            else:
                                args[key] = parsed_value
                                continue
                        except Exception as e:
                            pass
                    args[key] = value

                # Plot function calls
                if statement['function'] == 'LINEPLOT':
                    result = plot_line(args.get('data', []),
                                       args.get('x_label', ''),
                                       args.get('y_label', ''),
                                       args.get('title'),
                                       args.get('legend'))
                elif statement['function'] == 'histogram':
                    result = plot_histogram(args.get('data', []),
                                            args.get('x_label', ''),
                                            args.get('y_label', ''),
                                            int(args.get('bins', 10)),
                                            args.get('title'))
                elif statement['function'] == 'scatter':
                    result = plot_scatter(args.get('x_data', []),
                                          args.get('y_data', []),
                                          args.get('x_label', ''),
                                          args.get('y_label', ''),
                                          args.get('title'))
                elif statement['function'] == 'bar':
                    result = plot_bar(args.get('categories', []),
                                      args.get('values', []),
                                      args.get('x_label', ''),
                                      args.get('y_label', ''),
                                      args.get('orientation', 'vertical'),
                                      args.get('title'))
                else:
                    result = {"status": "error", "message": f"Unknown plot type: {statement['function']}"}

            elif stmt_type == 'Export':
                # Handle both 'table' and variable references
                identifier = statement.get('table')
                if identifier and identifier.startswith('$'):
                    # It's already a variable reference
                    pass
                elif 'variable' in statement:
                    identifier = '$' + statement['variable']

                result = export_statement(identifier, statement['column'], statement['to'])

            elif stmt_type == 'Loop':
                dictionary = {}
                for _ in range(int(statement['from']), int(statement['to']) + 1):
                    dictionary['statements'] = statement['body']
                    interpret(dictionary)
                result = {"status": "success",
                          "message": f"Loop executed from {statement['from']} to {statement['to']}"}

            elif stmt_type == 'Clean':
                if 'column' in statement:
                    if '.' in statement['column']:
                        table_name, column_name = statement['column'].split('.')
                    else:
                        table_name = statement.get('table', '')
                        column_name = statement['column']

                    action = statement.get('action', '').lower()

                    if action == 'remove':
                        result = clean_column(table_name, column_name, "delete")
                    elif action == 'replace':
                        result = clean_column(table_name, column_name, "replace", statement.get('replaceWith'))
                    else:
                        result = {"status": "error", "message": f"Unknown clean action: {action}"}
                else:
                    result = {"status": "error", "message": "Missing column in Clean statement"}
            else:
                result = {"status": "error", "message": f"Unknown statement type: {stmt_type}"}

        except Exception as e:
            result = {"status": "error", "message": f"Error executing {stmt_type}: {str(e)}",
                      "traceback": traceback.format_exc()}

        results.append(result)

    return {"status": "success", "results": results}


def execute_chronolang_json(json_input):
    """
    Main function to execute ChronoLang code from JSON input.

    Args:
        json_input: Either a JSON string or a dictionary containing:
            - 'code': ChronoLang source code to parse and execute
            - OR 'ast': Pre-parsed AST to execute directly

    Returns:
        dict: JSON response with execution results
    """
    try:
        # Parse JSON input if it's a string
        if isinstance(json_input, str):
            request_data = json.loads(json_input)
        else:
            request_data = json_input

        # Check if it's raw ChronoLang code or already parsed AST
        if 'code' in request_data:
            # Parse ChronoLang code
            src = request_data['code']
            json_str = chrono.chrono_parse(src.encode('utf-8'))
            ast = json.loads(json_str)
            print(ast)
        elif 'ast' in request_data:
            # Use provided AST directly
            ast = request_data['ast']
        else:
            return {"status": "error", "message": "Either 'code' or 'ast' must be provided"}

        # Execute the AST
        result = interpret(ast)
        return result

    except Exception as e:
        return {
            "status": "error",
            "message": f"Execution failed: {str(e)}",
            "traceback": traceback.format_exc()
        }


def get_datasets_info():
    """Get information about loaded datasets"""
    try:
        dataset_info = {}
        for name, df in datasets.items():
            if hasattr(df, 'columns'):
                dataset_info[name] = {
                    "columns": list(df.columns),
                    "shape": df.shape,
                    "dtypes": df.dtypes.to_dict()
                }
            else:
                dataset_info[name] = {"type": type(df).__name__, "value": str(df)[:100]}

        return {"status": "success", "datasets": dataset_info}

    except Exception as e:
        return {"status": "error", "message": f"Failed to get datasets: {str(e)}"}

#Example usage (for testing)
# if __name__ == '__main__':
#     # Example of how to use the functions

#     # Test with ChronoLang code
#     test_code = {
#         "code": """LOAD sales_data FROM "Amazon.csv"

#         TREND(sales_data.Open) -> forecast_next(7d)


#         SELECT sales_data.Low WHERE DATE < "2019-01-01" AS var_1
#         REMOVE missing FROM sales_data.Low
#         EXPORT sales_data.Low TO "../results/run3.csv"

#         FORECAST $var_1 USING Prophet(model_order=3, seasonal_order=2)
#         PLOT histogram(
#             data=[$var_1],
#             x_label="Days",
#             y_label="Sales",
#             bins=10,
#             title="Weekly Sales"
#         )
# """
#     }

#     result = execute_chronolang_json(test_code)
#     datasets_info = get_datasets_info()
#     print(result)
#     print(datasets_info)