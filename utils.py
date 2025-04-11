"""
Utility functions for data processing and manipulation.
"""
import json
import logging
import os
import pandas as pd
from ast import literal_eval

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_directories_exist(directories):
    """
    Ensure that all specified directories exist, creating them if necessary.
    
    Args:
        directories (list): List of directory paths to create.
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def format_arguments(arguments):
    """
    Format function arguments into a string suitable for a function call.
    
    Args:
        arguments (dict): Dictionary of argument names and values.
        
    Returns:
        str: Formatted argument string.
    """
    formatted_args = []
    for key, value in arguments.items():
        if isinstance(value, str):
            formatted_value = "'" + value.replace("'", r"\'") + "'"
        elif isinstance(value, bool):
            formatted_value = str(value)
        elif isinstance(value, (int, float)):
            formatted_value = str(value)
        elif value is None:
            formatted_value = 'None'
        else:
            formatted_value = value
        formatted_args.append(f"{key}={formatted_value}")
    return ", ".join(formatted_args)

def merge_csv_files(file_paths, output_path):
    """
    Merge multiple CSV files into a single CSV file.
    
    Args:
        file_paths (list): List of CSV file paths to merge.
        output_path (str): Path to save the merged CSV file.
        
    Returns:
        int: Number of rows in the merged file, or None if an error occurred.
    """
    try:
        # Load and merge CSV files
        df_list = [pd.read_csv(file) for file in file_paths]
        merged_df = pd.concat(df_list, ignore_index=True)
        merged_df.to_csv(output_path, index=False)
        logger.info(f"Merged file saved at {output_path} with {merged_df.shape[0]} rows.")
        return merged_df.shape[0]
    except FileNotFoundError as e:
        logger.error(f"Error merging CSV files: {e}")
        return None

def load_json_safely(json_str):
    """
    Safely load a JSON string, falling back to literal_eval if needed.
    
    Args:
        json_str (str): JSON string to parse.
        
    Returns:
        dict or list: Parsed JSON object.
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            return literal_eval(json_str)
        except (ValueError, SyntaxError) as e:
            logger.error(f"Failed to parse JSON string: {e}")
            return None 