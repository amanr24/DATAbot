'''
python_run.py

Utility functions to extract and execute Python code snippets from LLM responses,
automatically install missing dependencies, and run visualization routines.
'''

import re
import runpy
import os
import importlib.util
import subprocess
import sys


def install_missing_libraries(code: str):
    '''
    Parse import statements in the given code and install any missing libraries.

    Args:
        code (str): Python source containing import statements.

    Side effects:
        Calls pip install for each missing package.
    '''
    # Find top-level modules from import statements
    import_statements = re.findall(r"^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)", code, re.MULTILINE)
    missing = []
    for module in import_statements:
        pkg = module.split('.')[0]
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)

    # Install each unique missing package
    for pkg in set(missing):
        try:
            print(f"Installing missing library: {pkg}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])
        except Exception as e:
            print(f"Failed to install {pkg}: {e}")
    if not missing:
        print("All required libraries are already installed.")


def extract_python_code(response: str, attempt: int) -> str:
    '''
    Extract the first Python code block from an LLM response.

    Args:
        response (str): LLM output containing ```python or ``` delimited code.
        attempt (int): 1 for first attempt (returns placeholder on failure), >1 raises on failure.

    Returns:
        str: The code inside the backticks or a placeholder if not found.

    Raises:
        ValueError: If no code found and attempt > 1.
    '''
    # Look for ```python and generic ``` code fences
    match_py = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    match_generic = re.search(r"```\n(.*?)```", response, re.DOTALL)
    if match_py:
        return match_py.group(1)
    if match_generic:
        return match_generic.group(1)
    if attempt == 1:
        return "NO PYTHON CODE FOUND"
    raise ValueError("NO PYTHON CODE FOUND")


def execute_visualizations(code: str, temp_file: str = 'temp_visualization.py'):
    '''
    Execute visualization code in an isolated module and return plots.

    Steps:
    1. Install missing libraries.
    2. Write code to a temporary file.
    3. Dynamically import and execute create_visualizations().
    4. Clean up the temp file.

    Args:
        code (str): Python code defining create_visualizations().
        temp_file (str): Path to write the temp module.

    Returns:
        list or str: List of visualization objects if successful, else error string.
    '''
    try:
        # Ensure dependencies
        install_missing_libraries(code)

        # Write to file
        with open(temp_file, 'w') as f:
            f.write(code)

        # Load module dynamically
        spec = importlib.util.spec_from_file_location('temp_visualization', temp_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Execute create_visualizations()
        if hasattr(module, 'create_visualizations'):
            return module.create_visualizations()
        raise AttributeError("No function 'create_visualizations' found.")
    except Exception as e:
        return f"Error executing the code: {e}"
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)
