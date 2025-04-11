"""
Process the tool_shuffle_small dataset from Hugging Face.
This script downloads the dataset, processes it, and saves it in a modified format.
"""
import json
import pandas as pd
import logging
import os
from datasets import load_dataset

# Import from project root
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOOL_SHUFFLE_SMALL_DIR, ORIGIN_FILE_PATH, MODIFIED_FILE_PATH
from utils import format_arguments, load_json_safely, logger

def huggingface_loader(dataset_name, split="train", name=None):
    """
    Load a dataset from Hugging Face and save it as a CSV file.
    
    Args:
        dataset_name (str): Name of the dataset to load.
        split (str): Dataset split to load.
        name (str, optional): Dataset configuration name.
        
    Returns:
        Dataset: Loaded dataset.
    """
    logger.debug(f"Loading {dataset_name}")
    
    logger.debug("Loading from web ...")
    ds = load_dataset(dataset_name, split=split, name=name, token=os.getenv("HF_TOKEN", None))
    # Save data as CSV
    df = pd.DataFrame(ds)
    df.to_csv(ORIGIN_FILE_PATH, index=False)
    logger.debug(f"Loaded and saved to {ORIGIN_FILE_PATH}")
    return ds

def process_conversation(conversation):
    """
    Process a conversation to format tool calls and remove tool call entries.
    
    Args:
        conversation (list): List of conversation entries.
        
    Returns:
        list: Processed conversation with formatted tool calls.
    """
    # Track tool calls to modify subsequent assistant messages
    for i in range(len(conversation)):
        try:
            entry = conversation[i]
            if entry['role'] == 'tool call':
                content = entry['content']
                name = content.get('name', '')
                arguments = content.get('arguments', {})
                # Update the next assistant entry if present
                if i + 1 < len(conversation) and conversation[i+1]['role'] == 'assistant':
                    assistant_entry = conversation[i+1]
                    args_str = format_arguments(arguments)
                    function_call = f"{name}({args_str})" if args_str else f"{name}()"
                    assistant_entry['content'] = function_call
        except Exception as e:
            logger.error(f"Error processing conversation entry: {e}")
    
    # Remove all entries where role is 'tool call'
    return [entry for entry in conversation if entry['role'] != 'tool call']

def process_dataframe(df):
    """
    Process each row in the DataFrame to format conversations and tools.
    
    Args:
        df (DataFrame): DataFrame to process.
        
    Returns:
        DataFrame: Processed DataFrame.
    """
    for index, row in df.iterrows():
        # Parse the conversation and tools
        conversation = load_json_safely(row['conversation'])
        tools = load_json_safely(row['tools'])
        
        if conversation is None or tools is None:
            logger.warning(f"Skipping row {index} due to invalid JSON")
            continue
        
        # Process the conversation
        processed_conversation = process_conversation(conversation)
        
        # Update the DataFrame
        df.at[index, 'tools'] = json.dumps(tools)
        df.at[index, 'conversation'] = json.dumps(processed_conversation)
    
    return df

def main():
    """Main function to process the tool_shuffle_small dataset."""
    # Load the dataset
    huggingface_loader("BitAgent/tool_shuffle_small")
    
    # Read the CSV file
    df = pd.read_csv(ORIGIN_FILE_PATH)
    
    # Process the DataFrame
    processed_df = process_dataframe(df)
    
    # Save the modified DataFrame
    processed_df.to_csv(MODIFIED_FILE_PATH, index=False)
    logger.info(f"Processed data saved to {MODIFIED_FILE_PATH}")

if __name__ == "__main__":
    main()