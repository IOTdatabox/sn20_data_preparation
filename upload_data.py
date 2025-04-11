"""
Upload processed dataset to Hugging Face Hub.
"""
import os
import json
import logging
from huggingface_hub import login
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict

# Import from project root
from config import (
    MODEL_NAME, 
    MAX_SEQ_LENGTH, 
    HF_TOKEN, 
    HF_USERNAME
)
from utils import logger

def load_dataset_from_file(dataset_path):
    """
    Load a dataset from a CSV file.
    
    Args:
        dataset_path (str): Path to the CSV file.
        
    Returns:
        Dataset: Loaded dataset.
    """
    logger.info(f"Loading dataset from {dataset_path}")
    return load_dataset('csv', data_files=dataset_path)['train']

def tokenize_dataset(dataset, tokenizer):
    """
    Tokenize the dataset using the model's tokenizer.
    
    Args:
        dataset (Dataset): Dataset to tokenize.
        tokenizer (AutoTokenizer): Tokenizer to use.
        
    Returns:
        Dataset: Tokenized dataset.
    """
    logger.info("Tokenizing dataset")
    
    def tokenize_function(examples):
        """
        Tokenize examples using the model's tokenizer.
        
        Args:
            examples (dict): Dictionary of examples.
            
        Returns:
            dict: Dictionary with tokenized text.
        """
        batch = {
            "text": []
        }
        for conv_str, tools_str in zip(examples['conversation'], examples['tools']):
            try:
                # Parse JSON columns
                conversation = json.loads(conv_str)
                tools = json.loads(tools_str)
                full_tokenized = tokenizer.apply_chat_template(
                    conversation,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=False
                )
                batch["text"].append(full_tokenized)
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Skipping invalid row: {e}")
                continue
        return batch
    
    # Process dataset in batches
    return dataset.map(tokenize_function, batched=True, batch_size=2, remove_columns=dataset.column_names)

def filter_empty_strings(dataset):
    """
    Filter out empty strings from the dataset.
    
    Args:
        dataset (Dataset): Dataset to filter.
        
    Returns:
        Dataset: Filtered dataset.
    """
    logger.info("Filtering empty strings")
    return dataset.filter(lambda x: len(x['text'].strip()) > 0)

def split_dataset(dataset, test_size=0.2):
    """
    Split the dataset into train and test sets.
    
    Args:
        dataset (Dataset): Dataset to split.
        test_size (float): Proportion of the dataset to include in the test split.
        
    Returns:
        DatasetDict: Dictionary containing train and test datasets.
    """
    logger.info(f"Splitting dataset with test_size={test_size}")
    train_test_split = dataset.train_test_split(test_size=test_size)
    
    # Wrap in DatasetDict for Hugging Face Hub compatibility
    return DatasetDict({
        "train": train_test_split["train"],
        "test": train_test_split["test"]
    })

def push_to_hub(dataset_dict, repo_id):
    """
    Push the dataset to the Hugging Face Hub.
    
    Args:
        dataset_dict (DatasetDict): Dataset to push.
        repo_id (str): Repository ID on Hugging Face Hub.
    """
    logger.info(f"Pushing dataset to {repo_id}")
    dataset_dict.push_to_hub(repo_id)
    logger.info("Dataset pushed successfully!")

def main():
    """
    Main function to upload the dataset to Hugging Face Hub.
    """
    # Configuration
    DATASET_PATH = "../bitAgent.csv"
    REPO_ID = f"{HF_USERNAME}/mistral-7b-instruct-templated-dataset"
    
    # Login to Hugging Face
    login(token=HF_TOKEN)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load and process dataset
    dataset = load_dataset_from_file(DATASET_PATH)
    formatted_ds = tokenize_dataset(dataset, tokenizer)
    filtered_ds = filter_empty_strings(formatted_ds)
    dataset_dict = split_dataset(filtered_ds)
    
    # Push to Hugging Face Hub
    push_to_hub(dataset_dict, REPO_ID)

if __name__ == "__main__":
    main()
