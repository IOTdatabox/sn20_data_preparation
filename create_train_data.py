"""
Create training data by processing and tokenizing the dataset.
"""
import os
import logging
import json
import random
import pandas as pd
from huggingface_hub import login
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict

# Import from project root
from config import (
    MODEL_NAME, 
    TOOL_SHUFFLE_SMALL_DIR, 
    TRAIN_OUTPUT_DIR, 
    MODIFIED_FILE_PATH, 
    MERGED_FILE_PATH, 
    COMPLETED_FILE_PATH,
    TRAIN_FILE_PATH,
    TEST_FILE_PATH,
    SYSTEM_PROMPT,
    HF_TOKEN
)
from utils import ensure_directories_exist, merge_csv_files, logger
import data_preprocessing.tool_shuffle_small_processed as tool_shuffle

def execute_preprocessing_scripts():
    """
    Execute preprocessing scripts to prepare the data.
    """
    try:
        logger.info("Executing preprocessing script: tool_shuffle_small_processed.py")
        tool_shuffle.main()
        logger.info("All data processing steps completed successfully.")
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

def add_random_tools_to_dataset():
    """
    Add random tools to the dataset to create a more challenging training set.
    
    Returns:
        int: Number of rows with errors.
    """
    # Read the merged dataset
    modified_df = pd.read_csv(MERGED_FILE_PATH)
    
    # Shuffle the dataset
    modified_df = modified_df.sample(frac=1, random_state=572343).reset_index(drop=True)
    
    # Collect all tools from all rows
    all_tools = []
    for tools_str in modified_df['tools']:
        tools = json.loads(tools_str)
        all_tools.extend(tools)
    
    # Read the merged dataset again
    merged_df = pd.read_csv(MERGED_FILE_PATH)
    
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
            tools_list = json.loads(tools_str)
            
            # Extract existing tool names for comparison
            existing_tool_names = set()
            for tool in tools_list:
                if isinstance(tool, dict) and 'name' in tool:
                    existing_tool_names.add(tool['name'])
            
            # Determine the number of tools to add (2-4)
            num_tools_to_add = random.randint(2, 4)
            
            # Filter available tools to only include those not already in the list
            available_tools = [t for t in all_tools if isinstance(t, dict) and 'name' in t and t['name'] not in existing_tool_names]
            
            # If we have available unique tools, add them
            added_count = 0
            while added_count < num_tools_to_add and available_tools:
                # Select a random tool from available tools
                selected_tool = random.choice(available_tools)
                
                # Add the tool to the list
                if random.choice([True, False]):
                    tools_list.append(selected_tool)
                else:
                    tools_list.insert(0, selected_tool)
                
                # Remove the selected tool from available tools to prevent duplicates
                available_tools = [t for t in available_tools if t['name'] != selected_tool['name']]
                
                # Increment counter
                added_count += 1
            
            return json.dumps(tools_list)
        except Exception as e:
            errored_rows += 1
            logger.error(f"Error updating tools: {e}")
            return tools_str
    
    merged_df['tools'] = merged_df['tools'].apply(update_tools)
    
    # Save the modified dataset
    merged_df.to_csv(COMPLETED_FILE_PATH, index=False)
    logger.info(f"Successfully added random tools to {COMPLETED_FILE_PATH}. Errored rows: {errored_rows}")
    
    return errored_rows

def tokenize_dataset():
    # Login to Hugging Face
    login(token=HF_TOKEN)
    
    # Load and verify dataset
    dataset = load_dataset('csv', data_files=COMPLETED_FILE_PATH)['train']
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):

        batch = {"text": []}
        for conv_str, tools_str in zip(examples['conversation'], examples['tools']):
            try:
                # Parse JSON columns
                conversation = json.loads(conv_str)
                tools = json.loads(tools_str)
                
                # Construct system message with tools
                system_content = SYSTEM_PROMPT + f"\n{json.dumps(tools, indent=2)}"
                formatted_conversation = [{"role": "system", "content": system_content}]
                
                # Extract first user message as query
                user_messages = [m for m in conversation if m.get('role') == 'user']
                if user_messages:
                    formatted_conversation.append({"role": "user", "content": user_messages[0].get('content', '')})
                
                # Add assistant messages
                for message in conversation:
                    if message.get('role') == 'assistant':
                        content = message.get('content', '')
                        if content == '()':
                            content = ''
                        formatted_conversation.append({"role": "assistant", "content": content})
                
                # Use the built-in chat template
                full_tokenized = tokenizer.apply_chat_template(
                    formatted_conversation,
                    tokenize=False,
                    add_generation_prompt=False
                )
                
                batch["text"].append(full_tokenized)
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Skipping invalid row: {e}")
                continue
        return batch
    
    # Process dataset in batches
    formatted_ds = dataset.map(tokenize_function, batched=True, batch_size=2, remove_columns=dataset.column_names)
    
    return formatted_ds


def save_tokenized_dataset(formatted_ds):

    filtered_ds = formatted_ds.filter(lambda x: len(x['text'].strip()) > 0)

    # Split dataset into train (80%) and test (20%)
    train_test_split = filtered_ds.train_test_split(test_size=0.2)

    # Wrap in DatasetDict for Hugging Face Hub compatibility
    dataset_dict = DatasetDict({
        "train": train_test_split["train"],
        "test": train_test_split["test"]
    })
    print('dataset_dict', dataset_dict)

    # Save train and test datasets as CSV files
    train_df = pd.DataFrame(dataset_dict["train"])
    test_df = pd.DataFrame(dataset_dict["test"])

    train_df.to_csv(TRAIN_FILE_PATH, index=False)
    test_df.to_csv(TEST_FILE_PATH, index=False)

    print(f"Datasets saved successfully to {TRAIN_OUTPUT_DIR}!")

def main():
    """
    Main function to create the training dataset.
    """
    # Ensure directories exist
    ensure_directories_exist([TOOL_SHUFFLE_SMALL_DIR, TRAIN_OUTPUT_DIR])
    
    # Execute preprocessing scripts
    execute_preprocessing_scripts()
    
    # Merge files
    num_rows = merge_csv_files([MODIFIED_FILE_PATH], MERGED_FILE_PATH)
    
    if num_rows is not None:
        logger.info(f"Merged file saved at {MERGED_FILE_PATH} with {num_rows} rows.")
        
        # Add random tools to the dataset
        errored_rows = add_random_tools_to_dataset()
        
        # Tokenize the dataset
        formatted_ds = tokenize_dataset()
        
        # Save the tokenized dataset
        save_tokenized_dataset(formatted_ds)

if __name__ == "__main__":
    main()

