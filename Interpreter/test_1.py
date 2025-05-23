import subprocess
import json
import os
import sys
import tempfile

def run_chronolang_json(code_or_json):
    """
    Run the ChronoLang interpreter executable with the given code or JSON
    
    Args:
        code_or_json: Either a string with ChronoLang code or a dict with {"code": "..."}
    
    Returns:
        dict: The parsed JSON result from the interpreter
    """
    # Ensure we have proper JSON input
    if isinstance(code_or_json, str):
        input_json = {"code": code_or_json}
    else:
        input_json = code_or_json
    
    # Path to the executable (in the same directory as this script)
    exe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                           "dist", "chronolang_interpreter.exe")
    
    if not os.path.exists(exe_path):
        # Try current directory
        exe_path = "chronolang_interpreter.exe"
        if not os.path.exists(exe_path):
            raise FileNotFoundError(f"Could not find chronolang_interpreter.exe")
    
    # Create a temporary file for input
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as input_file:
        input_path = input_file.name
        json.dump(input_json, input_file)
    
    # Create a temporary file for output
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as output_file:
        output_path = output_file.name
    
    try:
        # Run the executable with input and output file paths as arguments
        process = subprocess.Popen(
            [exe_path, input_path, output_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for the process to complete
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"Interpreter failed with error: {stderr}")
        
        # Read the output file
        with open(output_path, 'r') as f:
            try:
                result = json.load(f)
                return result
            except json.JSONDecodeError:
                # If not valid JSON, return raw output
                f.seek(0)
                return {"status": "error", "raw_output": f.read(), "stderr": stderr}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
    finally:
        # Clean up temporary files
        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


# Example usage
if __name__ == "__main__":
    test_code = """
    LOAD sales_data FROM "Amazon.csv"

    TREND(sales_data.Open) -> forecast_next(7d)

    SELECT sales_data.Volume WHERE DATE == "2019-01-01" 
    REMOVE missing FROM sales_data.Low
    EXPORT sales_data.Low TO "results/run.csv"

    PLOT LINEPLOT(
        data=[sales_data.High,sales_data.Low],
        x_label="Days",
        y_label="Sales",
        title="Weekly Sales",
        legend=["1", "2"]
    )
    FOR i IN 1 TO 3 {
        FORECAST sales_data.Open USING Prophet(model_order=3, seasonal_order=2)
        EXPORT sales_data.Open TO "results/run1.csv"
    }
    """
    
    result = run_chronolang_json(test_code)
    print(json.dumps(result, indent=2))