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
import traceback

# Load the ChronoLang DLL
dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'chronolang.dll'))
chrono = cdll.LoadLibrary(dll_path)
chrono.chrono_parse.argtypes = [c_char_p]
chrono.chrono_parse.restype = c_char_p

datasets = {}

def load_statement(identifier: str, data_source: str):
    """Loads data from a CSV file into a dictionary."""
    try:
        datasets[identifier] = pd.read_csv(data_source)
        return {"status": "success", "message": f"Loaded {identifier} from {data_source}"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to load {identifier}: {str(e)}"}

def export_statement(identifier: str, column: str, destination: str):
    """Exports a dataset column or a forecast result to a CSV file."""
    
    if identifier not in datasets:
        return {"status": "error", "message": f"Dataset '{identifier}' not found."}
    
    data = datasets[identifier]
    
    if column not in data:
        return {"status": "error", "message": f"Column '{column}' not found in dataset '{identifier}'."}

    try:
        df = pd.DataFrame({column: data[column]})
        df.to_csv(destination, index=False)
        return {"status": "success", "message": f"Exported column '{column}' from dataset '{identifier}' to {destination}"}
    except Exception as e:
        return {"status": "error", "message": f"Export failed: {str(e)}"}

def set_statement(var_name: str, value):
    """Stores a variable in the interpreter's memory."""
    try:
        datasets[var_name] = value
        return {"status": "success", "message": f"Variable {var_name} set successfully."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to set variable: {str(e)}"}

def select_statement(identifier: str, column: str, op: str = None, date_expr: str = None):
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
                    return {"status": "error", "message": f"No date column found in dataset {identifier} for comparison."}
            
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
            
            return {
                "status": "success",
                "message": f"Selected {len(result_data)} rows from {column} where {comparison_column} {op} {date_expr}",
                "data": result_data
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Selection error: {str(e)}"}
    else:
        # No condition provided, return the entire column
        return {"status": "success", "message": "No condition applied.", "data": df[column].to_dict()}

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

def transform_statement(identifier:str, column: str, time_interval: str):
    """Applies a simple linear trend to forecast the next value."""
    try:
        if identifier in datasets:
            df = datasets[identifier]
            x = np.arange(len(df))
            y = df[column]
            coef = np.polyfit(x, y, 1)
            forecast_value = coef[0] * (len(df) + int(time_interval)) + coef[1]
            return {
                "status": "success",
                "message": f"Forecasted next {time_interval} value for {column}", 
                "forecast": float(forecast_value)
            }
        return {"status": "error", "message": f"Data from {identifier} or column {column} not found in the {identifier}."}
    except Exception as e:
        return {"status": "error", "message": f"Transform failed: {str(e)}"}

def forecast_statement(identifier: str, column: str, model: str, params: dict):
    """Performs forecasting using ARIMA, Prophet, or LSTM."""
    
    if identifier not in datasets:
        return {"status": "error", "message": f"Dataset {identifier} not found."}
    
    df = datasets[identifier]

    if column not in df.columns:
        return {"status": "error", "message": f"Column '{column}' not found in dataset {identifier}."}

    if len(df) < 10:
        return {"status": "error", "message": "Not enough data for forecasting."}

    try:
        if model == "ARIMA":
            order = params.get("order", (1, 1, 1))
            model_fit = ARIMA(df[column], order=order).fit()
            forecast = model_fit.forecast(steps=params.get("steps", 1)).tolist()
            return {
                "status": "success",
                "message": f"Forecasted {column} using ARIMA with order {order}", 
                "forecast": forecast
            }

        elif model == "Prophet":
            if "Date" not in df.columns:
                return {"status": "error", "message": "Prophet requires a 'Date' column."}
            
            prophet_df = df.rename(columns={"Date": "ds", column: "y"})[["ds", "y"]]
            prophet = Prophet()
            prophet.fit(prophet_df)
            
            future = prophet.make_future_dataframe(periods=params.get("steps", 1))
            forecast = prophet.predict(future)[["ds", "yhat"]].tail(params.get("steps", 1)).to_dict(orient="records")
            return {"status": "success", "message": f"Forecasted {column} using Prophet.", "forecast": forecast}

        elif model == "LSTM":
            steps = params.get("steps", 1)
            look_back = params.get("look_back", 5)
            layers = params.get("layers", 3)

            data = df[column].values.reshape(-1, 1)
            x_train, y_train = [], []

            for i in range(len(data) - look_back):
                x_train.append(data[i:i+look_back])
                y_train.append(data[i+look_back])

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
            model.fit(x_train, y_train, epochs=50, verbose=0)

            x_input = data[-look_back:].reshape(1, look_back, 1)
            forecast = [float(model.predict(x_input, verbose=0)[0][0]) for _ in range(steps)]
            
            return {"status": "success", "message": f"Forecasted {column} using LSTM with {layers} layers.", "forecast": forecast}
        else:
            return {"status": "error", "message": f"Unsupported model type: {model}"}

    except Exception as e:
        return {"status": "error", "message": f"Forecasting error: {str(e)}"}

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
                    plt.plot(series, label=f'Series {i+1}')
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
                    for i in range(data_array.shape[1] if data_array.shape[0] < data_array.shape[1] else data_array.shape[0]):
                        plt.plot(data_array[i] if data_array.shape[0] > data_array.shape[1] else data_array[:, i], 
                                label=f'Series {i+1}')
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
    #plt.show()
    
    encoded_image = generate_plot()
    return {"status": "success", "message": "Line plot generated successfully.", "plot": encoded_image}

def plot_histogram(data, x_label: str, y_label: str, bins: int, title=None):
    """Generates a histogram."""
    try:
        plt.figure()
        plt.hist(data, bins=bins)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if title:
            plt.title(title)
        encoded_image = generate_plot()
        return {"status": "success", "message": "Histogram generated.", "plot": encoded_image}
    except Exception as e:
        return {"status": "error", "message": f"Histogram generation failed: {str(e)}"}

def plot_scatter(x_data, y_data, x_label: str, y_label: str, title=None):
    """Generates a scatter plot."""
    try:
        plt.figure()
        plt.scatter(x_data, y_data)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if title:
            plt.title(title)
        encoded_image = generate_plot()
        return {"status": "success", "message": "Scatter plot generated.", "plot": encoded_image}
    except Exception as e:
        return {"status": "error", "message": f"Scatter plot generation failed: {str(e)}"}

def plot_bar(categories, values, x_label: str, y_label: str, orientation: str, title=None):
    """Generates a bar plot."""
    try:
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
                result = transform_statement(statement['table'], statement['column'], statement['interval']['amount'])
                
            elif stmt_type == 'Forecast':
                params = {}
                for key, value in statement['params'].items():
                    params[key] = value
                result = forecast_statement(statement['table'], statement['column'], statement['model'], params)
            
            elif stmt_type == 'Stream':
                result = stream_statement(statement['id'], statement['path'])
            
            elif stmt_type == 'Select':
                column_to_retrieve = statement['column']
                
                if 'condition' in statement and 'op' in statement['condition'] and 'date' in statement['condition']:
                    op = statement['condition']['op']
                    date_value = statement['condition']['date']
                    result = select_statement(statement['table'], column_to_retrieve, op, date_value)
                else:
                    result = select_statement(statement['table'], column_to_retrieve)
                    
                if 'data' in result:
                    import pandas as pd
                    selected_data = pd.DataFrame({column_to_retrieve: result['data'].values()})
                    datasets[f"{statement['table']}_selected"] = selected_data
            
            elif stmt_type == 'Plot':
                args = {}
                for key, value in statement['args'].items():
                    
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
                    
                    if isinstance(value, str) and '.' in value and value.count('.') == 1:
                        try:
                            dataset_name, column_name = value.split('.')
                            if dataset_name in datasets and column_name in datasets[dataset_name].columns:
                                args[key] = datasets[dataset_name][column_name].tolist()
                                continue
                        except Exception as e:
                            pass
                    
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
                result = export_statement(statement['table'], statement['column'], statement['to'])
            
            elif stmt_type == 'Loop':
                dictionary = {}
                for _ in range(int(statement['from']), int(statement['to']) + 1):
                    dictionary['statements'] = statement['body']
                    interpret(dictionary)
                result = {"status": "success", "message": f"Loop executed from {statement['from']} to {statement['to']}"}
            
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
            result = {"status": "error", "message": f"Error executing {stmt_type}: {str(e)}", "traceback": traceback.format_exc()}
        
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

# Example usage (for testing)
# if __name__ == '__main__':
#     # Example of how to use the functions
    
#     # Test with ChronoLang code
#     test_code = {
#         "code": """LOAD sales_data FROM "Interpreter\Amazon.csv"

#         TREND(sales_data.Open) -> forecast_next(7d)
     

#         SELECT sales_data.Volume WHERE DATE == "2019-01-01" 
#         REMOVE missing FROM sales_data.Low
#         EXPORT sales_data.Low TO "results/run.csv"

#         PLOT LINEPLOT(
#             data=[sales_data.High,sales_data.Low],
#             x_label="Days",
#             y_label="Sales",
#             title="Weekly Sales",
#             legend=["1", "2"]
#         )
#         FOR i IN 1 TO 3 {
#             FORECAST sales_data.Open USING Prophet(model_order=3, seasonal_order=2)
#             EXPORT sales_data.Open TO "results/run1.csv"
#         }
# """
#     }
    
#     result = execute_chronolang_json(test_code)
#     datasets_info = get_datasets_info()
#     print(result)
#     print(datasets_info)