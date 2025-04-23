import re
import runpy
import os
import importlib.util
import subprocess
import sys


def install_missing_libraries(code):
    """
    Parses the Python code to identify missing libraries and installs them.
    """
    # Extract import statements from the code
    import_statements = re.findall(r"^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)", code, re.MULTILINE)
    missing_libraries = []

    for module in import_statements:
        # Check if the module is installed
        try:
            importlib.import_module(module.split('.')[0])  # Handle submodules like `pandas.core`
        except ImportError:
            missing_libraries.append(module.split('.')[0])

    # Install missing libraries
    if missing_libraries:
        for lib in set(missing_libraries):  # Remove duplicates
            try:
                print(f"Installing missing library: {lib}")
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            except Exception as e:
                print(f"Failed to install {lib}: {e}")
    else:
        print("All required libraries are already installed.")


def extract_python_code(response, attempt):
    """
    Extracts the Python code block from the LLM response.
    Assumes the code is enclosed within triple backticks (```).
    """
    code_block = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    code_block1 = re.search(r"```\n(.*?)```", response, re.DOTALL)
    if code_block:
        return code_block.group(1)
    elif code_block1:
        return code_block1.group(1)
    else:
        if attempt ==1:
            return "NO PYTHON CODE FOUND"
        else:
            raise ValueError("NO PYTHON CODE FOUND")


def execute_visualizations(code, temp_file="temp_visualization.py"):
    """
    Executes the extracted Python code in a separate file and
    returns the list of visualizations (figures).
    """
    try:
        # Check for missing libraries and install them
        install_missing_libraries(code)

        # Write the code to a temporary file
        with open(temp_file, "w") as file:
            file.write(code)

        # Dynamically load the module
        spec = importlib.util.spec_from_file_location("temp_visualization", temp_file)
        temp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(temp_module)

        # Call the create_visualizations function
        if hasattr(temp_module, "create_visualizations"):
            return temp_module.create_visualizations()
        else:
            raise AttributeError("No function 'create_visualizations' found in the provided code.")

    except Exception as e:
        return f"Error executing the code: {e}"

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
