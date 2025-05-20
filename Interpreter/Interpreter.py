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
from ast import literal_eval
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


    if identifier not in datasets:
        return {"error": f"Dataset {identifier} not found."}
    
    df = datasets[identifier]
    
    if column not in df.columns:
        return {"error": f"Column {column} not found in dataset {identifier}."}
    
    if op and date_expr:
        try:
            comparison_column = 'Date'
            
            if comparison_column not in df.columns:
                date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                if date_columns:
                    comparison_column = date_columns[0]
                else:
                    return {"error": f"No date column found in dataset {identifier} for comparison."}
            
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
                return {"error": f"Unsupported operator: {op}"}
            
            # Extract the requested column from the filtered data
            result_data = selected_data[column].to_dict() if not selected_data.empty else {}
            
            return {
                "message": f"Selected {len(result_data)} rows from {column} where {comparison_column} {op} {date_expr}"
                #"data": result_data
            }
            
        except Exception as e:
            return {"error": f"Selection error: {str(e)}"}
    else:
        # No condition provided, return the entire column
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
            layers = params.get("layers",3)

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

def safe_parse(value):
    if isinstance(value, str):
        try:
            # Import literal_eval at the top level and use it directly here
            import ast
            parsed = ast.literal_eval(value)
            return parsed
        except Exception:
            return value  # Return original string if it can't be parsed
    return value

def plot_line(data, x_label: str, y_label: str, title=None, legend=None):
    
    plt.figure(figsize=(10, 6))
    
    # Debug information
    print(f"Plot_line received data type: {type(data)}")
    print(f"Data content preview: {str(data)[:200]}...")
    
    # Handle the case where data is None or empty
    if data is None or (isinstance(data, list) and len(data) == 0):
        return {"error": "No data provided for plotting"}
    
    try:
        # Case 1: Single data series (list/array of values)
        if (isinstance(data, list) and all(isinstance(x, (int, float, np.number)) for x in data)) or \
           isinstance(data, np.ndarray) and data.ndim == 1:
            plt.plot(data, label='Series')
            use_legend = True
        
        # Case 2: List of data series
        elif isinstance(data, list) and all(isinstance(series, (list, np.ndarray)) for series in data):
            use_legend = False
            for i, series in enumerate(data):
                # Check if we have enough data points to plot
                if len(series) > 0:
                    plt.plot(series, label=f'Series {i+1}')
                    use_legend = True
        
        # Case 3: Pandas DataFrame or Series
        elif hasattr(data, 'plot'):
            data.plot(ax=plt.gca())
            use_legend = True
        
        # Case 4: Try to convert other types to a numpy array
        else:
            try:
                data_array = np.array(data)
                if data_array.ndim == 1:
                    plt.plot(data_array, label='Series')
                    use_legend = True
                elif data_array.ndim == 2:
                    for i in range(data_array.shape[1] if data_array.shape[0] < data_array.shape[1] else data_array.shape[0]):
                        plt.plot(data_array[i] if data_array.shape[0] > data_array.shape[1] else data_array[:, i], 
                                label=f'Series {i+1}')
                    use_legend = True
                else:
                    return {"error": "Data has too many dimensions for plotting"}
            except Exception as e:
                return {"error": f"Failed to convert data for plotting: {str(e)}"}
    
    except Exception as e:
        return {"error": f"Error during plot generation: {str(e)}"}
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)
    
    if legend:
        # Try to parse legend if it's a string
        if isinstance(legend, str):
            try:
                import ast
                parsed_legend = ast.literal_eval(legend)
                legend = parsed_legend if isinstance(parsed_legend, list) else [legend]
            except Exception:
                legend = [legend]  # Keep as a single string if parsing fails
        
        plt.legend(legend)
    elif use_legend:
        plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    encoded_image = generate_plot()
    return {"message": "Line plot generated successfully.", "plot": encoded_image}

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

def safe_parse(value):
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed
        except Exception:
            return value  # Return original string if it can't be parsed
    return value

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
            params = {}
            for key, value in statement['params'].items():
                params[key] = value
            result = forecast_statement(statement['table'], statement['column'], statement['model'], params)
        
        elif stmt_type == 'Stream':
            result = stream_statement(statement['id'], statement['path'])
        
        elif stmt_type == 'Select':
            # The column in the SELECT statement refers to what we want to retrieve
            column_to_retrieve = statement['column']
            
            # The condition typically includes a DATE column and a comparison
            if 'condition' in statement and 'op' in statement['condition'] and 'date' in statement['condition']:
                op = statement['condition']['op']
                date_value = statement['condition']['date']
                
                # Pass these to the select_statement function
                result = select_statement(statement['table'], column_to_retrieve, op, date_value)
            else:
                # No condition - retrieve all values for the column
                result = select_statement(statement['table'], column_to_retrieve)
                
            # Store the result in datasets to make it available for other operations
            if 'data' in result:
                # Create a DataFrame from the selected data
                import pandas as pd
                selected_data = pd.DataFrame({column_to_retrieve: result['data'].values()})
                datasets[f"{statement['table']}_selected"] = selected_data
        
        elif stmt_type == 'Plot':
            args = {}
            for key, value in statement['args'].items():
                print(f"Processing argument {key} with value: {value}")
                
                # CASE 1: Handle comma-separated list of dataset references without brackets
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
                            print(f"Processed comma-separated references for {key}: {len(processed_list)} series")
                            continue
                        except Exception as e:
                            print(f"Error processing comma-separated references: {str(e)}")
                
                # CASE 2: Handle the specific pattern: data=[sales_data.Open]
                if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                    inner_content = value[1:-1].strip()
                    
                    # CASE 2.1: Handle comma-separated dataset references inside brackets
                    # Example: [sales_data.High,sales_data.Low]
                    if ',' in inner_content and all('.' in part.strip() for part in inner_content.split(',')):
                        try:
                            parts = [part.strip() for part in inner_content.split(',')]
                            processed_list = []
                            
                            for part in parts:
                                dataset_name, column_name = part.split('.')
                                if dataset_name in datasets and column_name in datasets[dataset_name].columns:
                                    processed_list.append(datasets[dataset_name][column_name].tolist())
                                else:
                                    # If any reference is invalid, abort this processing method
                                    raise ValueError(f"Invalid dataset reference: {part}")
                            
                            args[key] = processed_list
                            print(f"Processed bracketed comma-separated references for {key}: {len(processed_list)} series")
                            continue
                        except Exception as e:
                            print(f"Error processing bracketed comma-separated references: {str(e)}")
                    
                    # CASE 2.2: Handle single dataset reference in brackets
                    # Example: [sales_data.Open]
                    if '.' in inner_content and inner_content.count('.') == 1:
                        try:
                            dataset_name, column_name = inner_content.split('.')
                            if dataset_name in datasets and column_name in datasets[dataset_name].columns:
                                # Use the actual data from the dataset column as a single element in a list
                                args[key] = [datasets[dataset_name][column_name].tolist()]
                                print(f"Processed single bracketed reference for {key}")
                                continue
                        except Exception as e:
                            print(f"Error processing single bracketed reference: {str(e)}")
                
                # CASE 3: Handle direct dataset column references (e.g., "sales_data.Open")
                if isinstance(value, str) and '.' in value and value.count('.') == 1:
                    try:
                        dataset_name, column_name = value.split('.')
                        if dataset_name in datasets and column_name in datasets[dataset_name].columns:
                            # Use the actual data from the dataset column
                            args[key] = datasets[dataset_name][column_name].tolist()
                            print(f"Processed single column reference for {key}")
                            continue
                    except Exception as e:
                        print(f"Error processing single column reference: {str(e)}")
                
                # CASE 4: Try to parse as Python literal (for complex list structures)
                if isinstance(value, str):
                    try:
                        import ast as python_ast
                        parsed_value = python_ast.literal_eval(value)
                        
                        if isinstance(parsed_value, list):
                            processed_list = []
                            all_elements_processed = True
                            
                            for item in parsed_value:
                                if isinstance(item, str) and '.' in item and item.count('.') == 1:
                                    # Looks like a dataset reference
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
                                print(f"Processed complex list reference for {key}")
                                continue
                            else:
                                args[key] = parsed_value
                                continue
                        else:
                            # Not a list, use the parsed value as is
                            args[key] = parsed_value
                            continue
                    except Exception as e:
                        print(f"Error parsing as Python literal: {str(e)}")
                args[key] = value
            
            if statement['function'] == 'LINEPLOT':
                '''result = '''
                plot_line(args.get('data', []), 
                                args.get('x_label', ''), 
                                args.get('y_label', ''), 
                                args.get('title'), 
                                args.get('legend'))
            elif statement['function'] == 'histogram':
                '''result = '''
                plot_histogram(args.get('data', []), 
                                    args.get('x_label', ''), 
                                    args.get('y_label', ''), 
                                    int(args.get('bins', 10)), 
                                    args.get('title'))
            elif statement['function'] == 'scatter':
                '''result =''' 
                plot_scatter(args.get('x_data', []), 
                                args.get('y_data', []), 
                                args.get('x_label', ''), 
                                args.get('y_label', ''), 
                                args.get('title'))
            elif statement['function'] == 'bar':
                '''result =''' 
                plot_bar(args.get('categories', []), 
                            args.get('values', []), 
                            args.get('x_label', ''), 
                            args.get('y_label', ''), 
                            args.get('orientation', 'vertical'), 
                            args.get('title'))
            else:
                result = {"error": f"Unknown plot type: {statement['function']}"}
        elif stmt_type == 'Export':
            result = export_statement(statement['table'], statement['column'], statement['to'])
        
        elif stmt_type == 'Loop':
            dictionary = {}
            for _ in range(int(statement['from']), int(statement['to']) + 1):
                dictionary['statements'] = statement['body']
                interpret(dictionary)
        
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
                    result = {"error": f"Unknown clean action: {action}"}
            else:
                result = {"error": "Missing column in Clean statement"}
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