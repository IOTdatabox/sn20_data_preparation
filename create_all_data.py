import os
import logging
import pandas as pd
from itertools import islice
from datasets import load_dataset, load_from_disk
from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import ast
import json
import random
import time

MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
DATASET_LENGTH = 3600

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
HF_USERNAME = os.getenv('HF_USERNAME')
WANDB_TOKEN = os.getenv('WANDB_TOKEN')


class JSONDatasetIterator:
    def __init__(self):
        self.dataframes = []
        self.all_data = None
        self.index = 0

        # Load data from JSON files
        for filename in ["java", "javascript", "simple", "multiple", "sql", "live_simple", "live_multiple"]:
            bfcl_path = f"data-preprocessing/bitagent.data/bfcl/BFCL_v3_{filename}.json"
            bfcl_answer_path = f"data-preprocessing/bitagent.data/bfcl/possible_answer/BFCL_v3_{filename}.json"
            if os.path.exists(bfcl_path) and os.path.exists(bfcl_answer_path):
                df_data = pd.read_json(bfcl_path, lines=True)
                df_answer = pd.read_json(bfcl_answer_path, lines=True)
                df_data['ground_truth'] = df_answer['ground_truth']
                self.dataframes.append(df_data[['id', 'question', 'function', 'ground_truth']])
                print(f"Length of {filename} dataframe: {len(df_data)}")

        self.all_data = pd.concat(self.dataframes)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.all_data):
            row = self.all_data.iloc[self.index]
            self.index += 1
            return row
        else:
            raise StopIteration


def huggingface_loader(dataset_name, root_data_dir="data-preprocessing/bitagent.data", split="train", name=None):
    logger.debug(f"Loading {dataset_name}")
    dataset_dir = f"{root_data_dir}/{dataset_name.replace('/', '_')}"
    if os.path.exists(f"{dataset_dir}/state.json"):
        logger.debug(f"Loading from disk ({dataset_dir}) ...")
        ds = load_from_disk(dataset_dir)
    else:
        logger.debug("Loading from web ...")
        ds = load_dataset(dataset_name, split=split, name=name, token=os.getenv("HF_TOKEN", None))
        ds.save_to_disk(dataset_dir)
    logger.debug("Loaded.")
    return ds


def load_bfcl_dataset(dataset_name, root_data_dir="data-preprocessing/bitagent.data", split="train", name=None):
    snapshot_download(
        repo_id=dataset_name,
        allow_patterns="*.json",
        repo_type="dataset",
        local_dir="data-preprocessing/bitagent.data/bfcl/"
    )
    return JSONDatasetIterator()


def sample_and_save_datasets(output_dir="data-preprocessing/bitagent.data/samples", sample_size=1000):
    os.makedirs(output_dir, exist_ok=True)
    try:
        glaive_ds = huggingface_loader("glaiveai/glaive-function-calling-v2")
        glaive_df = pd.DataFrame(glaive_ds)
        #500 rows
        glaive_sample = glaive_df.sample(n=min(25000, len(glaive_df))) 
        # glaive_sample = glaive_df.sample(frac=1)
        glaive_sample.to_csv(f"{output_dir}/glaive_sample.csv", index=False)
        logger.info(f"Saved Glaive sample to {output_dir}/glaive_sample.csv")
    except Exception as e:
        logger.error(f"Error processing Glaive dataset: {str(e)}")

    try:
        bfcl_ds = load_bfcl_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard")
        # bfcl_df = pd.DataFrame(list(bfcl_ds))
        bfcl_sample = pd.DataFrame(list(islice(bfcl_ds, 1000)))
        # bfcl_sample = pd.DataFrame(list(bfcl_ds))
        bfcl_sample.to_csv(f"{output_dir}/bfcl_sample.csv", index=False)
        logger.info(f"Saved BFCL sample to {output_dir}/bfcl_sample.csv")
    except Exception as e:
        logger.error(f"Error processing BFCL dataset: {str(e)}")

    try:
        bitagent_ds = huggingface_loader("BitAgent/tool_calling_shuffle")
        bitagent_df = pd.DataFrame(bitagent_ds)
        bitagent_sample = bitagent_df.sample(n=min(4000, len(bitagent_df)))
        # bitagent_sample = bitagent_df.sample(frac=1)
        bitagent_sample.to_csv(f"{output_dir}/bitagent_sample.csv", index=False)
        logger.info(f"Saved BitAgent sample to {output_dir}/bitagent_sample.csv")
    except Exception as e:
        logger.error(f"Error processing BitAgent dataset: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())




if __name__ == "__main__":
    # Step 1: Download and sample datasets
    sample_and_save_datasets()



logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
try:
    processing_scripts = [
        'data-preprocessing/tool_calling_shuffle_processed.py',
        'data-preprocessing/bfcl_processed.py',
        'data-preprocessing/glaive_processed.py'
    ]
    
    for script in processing_scripts:
        logger.info(f"Executing processing script: {script}")
        with open(script) as file:
            exec(file.read())
    
    logger.info("All data processing steps completed successfully.")
except Exception as e:
    print(f"Error during preprocessing: {e}")
    exit()  # Stop execution if preprocessing fails



def merge_csv_files(file_paths, output_path):
    try:
        # Load and merge CSV files
        df_list = [pd.read_csv(file) for file in file_paths]
        merged_df = pd.concat(df_list, ignore_index=True)

        # Remove rows where either 'conversations' or 'tools' is empty
        merged_df = merged_df.dropna(subset=['conversation', 'tools'])
        merged_df = merged_df[
            (merged_df['conversation'].astype(str).str.strip() != '') &
            (merged_df['conversation'].astype(str) != '[]') &
            (merged_df['tools'].astype(str).str.strip() != '') &
            (merged_df['tools'].astype(str) != '[]')
        ]
        # Save the merged file
        merged_df.to_csv(output_path, index=False)

        # Return the number of rows in the merged file
        return merged_df.shape[0]
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

# File paths
file_paths = [
    "data-preprocessing/bitagent.data/samples/bitagent_modified.csv",
    "data-preprocessing/bitagent.data/samples/bfcl_modified.csv",
    "data-preprocessing/bitagent.data/samples/glaive_modified.csv"
]

# Output file path
merged_path = "bitAgent.csv"

# Merge files and get the number of rows
num_rows = merge_csv_files(file_paths, merged_path)

if num_rows is not None:
    print(f"Merged file saved at {merged_path} with {num_rows} rows.")


# ADD EXTRAB TOOLS IN TOOLS COLUMNS
# Read source CSV and extract tools
modified_path = "bitAgent.csv"
modified_df = pd.read_csv(modified_path)

# Collect all tools from all rows
all_tools = []
for tools_str in modified_df['tools']:
    # print('tools_str', tools_str)
    tools = json.loads(tools_str)
    all_tools.extend(tools)

# merged_path = "data-preprocessing/bitagent.data/samples/merged.csv"
merged_df = pd.read_csv(merged_path)

errored_rows = 0
# Update tools column with JSON-safe formatting
def update_tools(tools_str):
    try:
      global errored_rows
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
        print(f"Unexpected error: {e}", tools_str)
        return tools_str
  
# Update tools column with JSON-safe formatting
# def update_tools(tools_str):
#   try:
#       global errored_rows
#       selected_tool = random.choice(all_tools)
#       tools_list = json.loads(tools_str)
#       # Determine the number of tools to add (2-4)
#       num_tools_to_add = random.randint(2, 4)
#       print('num of tools to add',num_tools_to_add, 'existing tool length', len(tools_list) )
#       for _ in range(num_tools_to_add):
#           selected_tool = random.choice(all_tools)
#           # Randomly choose to append or prepend each tool
#           if random.choice([True, False]):
#              tools_list.append(selected_tool)
#           else:
#              tools_list.insert(0, selected_tool)
        
#       return json.dumps(tools_list)
#   except Exception as e:
#         errored_rows += 1
#         print(f"Unexpected error: {e}", tools_str)
#         return tools_str

merged_df['tools'] = merged_df['tools'].apply(update_tools)

# Save modified data back to merged.csv
    
merged_df.to_csv('bitAgentWithExtraTools.csv', index=False)

print(f"Successfully added tool name to merged.csv and these are errored rows", errored_rows)


# CREATE TEMPLATED DATA
import os
import json
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict

# Load environment variables from .env file
load_dotenv()

# Configuration
DATASET_PATH = "./bitAgentWithExtraTools.csv"
CSV_OUTPUT_DIR = "data/csv"

# Login to Hugging Face
login(token=os.getenv("HF_TOKEN"))

# Load and verify dataset
dataset = load_dataset('csv', data_files=DATASET_PATH)['train']

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    batch = {
        "text": []
    }
    for conv_str, tools_str in zip(examples['conversation'], examples['tools']):
        try:
            # Parse JSON columns
            conversation = json.loads(conv_str)
            tools = json.loads(tools_str)
            
            # Format conversation with tools in system message
            formatted_conversation = []
            
            # Add system message with tools
            system_content = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function call in tools call sections.

You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke:"""
            
            system_content += f"\n{json.dumps(tools, indent=2)}"
            
            # Add system message
            formatted_conversation.append({"role": "system", "content": system_content})
            
            # Add the rest of the conversation
            for message in conversation:
                formatted_conversation.append(message)
            
            # Apply chat template
            full_tokenized = tokenizer.apply_chat_template(
                formatted_conversation,
                tokenize=False,
                add_generation_prompt=False
            )
            
            batch["text"].append(full_tokenized)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skipping invalid row: {e}")
            continue
    return batch

# Process dataset in batches
formatted_ds = dataset.map(tokenize_function, batched=True, batch_size=2, remove_columns=dataset.column_names)
# print('formatted_ds', formatted_ds)  

# Filter empty strings
filtered_ds = formatted_ds.filter(lambda x: len(x['text'].strip()) > 0)

# Split dataset into train (80%) and test (20%)
train_test_split = filtered_ds.train_test_split(test_size=0.2)

# Wrap in DatasetDict for Hugging Face Hub compatibility
dataset_dict = DatasetDict({
    "train": train_test_split["train"],
    "test": train_test_split["test"]
})
print('dataset_dict', dataset_dict)

# Create output directory if it doesn't exist
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

# Save train and test datasets as CSV files
train_df = pd.DataFrame(dataset_dict["train"])
test_df = pd.DataFrame(dataset_dict["test"])

train_df.to_csv(os.path.join(CSV_OUTPUT_DIR, "train.csv"), index=False)
test_df.to_csv(os.path.join(CSV_OUTPUT_DIR, "test.csv"), index=False)

print(f"Datasets saved successfully to {CSV_OUTPUT_DIR}!")
