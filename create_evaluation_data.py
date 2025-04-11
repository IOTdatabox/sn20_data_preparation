"""
Create evaluation data by adding extra tool calls to the dataset.
"""
import os
import logging
import json
import random
import pandas as pd
from datasets import load_dataset, load_from_disk

# Import from project root
from config import (
    EVALUATION_OUTPUT_DIR, 
    TOOL_SHUFFLE_SMALL_DIR, 
    ORIGIN_FILE_PATH, 
    EVALUATION_FILE_PATH
)
from utils import ensure_directories_exist, logger

def add_extra_tool_calls():
    """
    Add extra tool calls to the dataset for evaluation purposes.
    
    This function reads the original dataset, collects all available tools,
    and adds random tools to each row to create a more challenging evaluation set.
    """
    # Ensure directories exist
    ensure_directories_exist([EVALUATION_OUTPUT_DIR, TOOL_SHUFFLE_SMALL_DIR])
    
    # Read the original dataset
    modified_df = pd.read_csv(ORIGIN_FILE_PATH)
    
    # Collect all tools from all rows
    all_tools = []
    for tools_str in modified_df['tools']:
        tools = json.loads(tools_str)
        all_tools.extend(tools)
    
    errored_rows = 0
    
    # Update tools column with JSON-safe formatting
    def update_tools(tools_str):
        """
        Update the tools string by adding random tools.
        
        Args:
            tools_str (str): JSON string of tools.
            
        Returns:
            str: Updated JSON string of tools.
        """
        try:
            nonlocal errored_rows
            selected_tool = random.choice(all_tools)
            tools_list = json.loads(tools_str)
            
            # Add 3 random tools
            for _ in range(3):
                selected_tool = random.choice(all_tools)
                tools_list.append(selected_tool)
                
            return json.dumps(tools_list)
        except Exception as e:
            errored_rows += 1
            logger.error(f"Error updating tools: {e}")
            return tools_str
    
    # Apply the update_tools function to each row
    modified_df['tools'] = modified_df['tools'].apply(update_tools)
    
    # Save the modified dataset
    modified_df.to_csv(EVALUATION_FILE_PATH, index=False)
    logger.info(f"Successfully added extra tool calls to {EVALUATION_FILE_PATH}. Errored rows: {errored_rows}")

def create_bitagent_shuffle_dataset():
    """
    Create the BitAgent shuffle dataset for evaluation.
    """
    try:
        add_extra_tool_calls()
    except Exception as e:
        logger.error(f"Error processing BitAgent dataset: {str(e)}")

if __name__ == "__main__":
    create_bitagent_shuffle_dataset()