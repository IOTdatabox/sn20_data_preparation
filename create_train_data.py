import os
import logging
import pandas as pd
from huggingface_hub import login
from transformers import AutoTokenizer
from datasets import load_dataset
from dotenv import load_dotenv
import json
import random

# Model and dataset configuration
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
DATASET_LENGTH = 3600

# Directory structure constants
TOOL_SHUFFLE_SMALL_DIR = "data/tool_shuffle_small"
TRAIN_OUTPUT_DIR = "data/train"

# File path constants
MODIFIED_FILE_PATH = os.path.join(TOOL_SHUFFLE_SMALL_DIR, "modified.csv")
MERGED_FILE_PATH = os.path.join(TOOL_SHUFFLE_SMALL_DIR, "merged.csv")
COMPLETED_FILE_PATH = os.path.join(TOOL_SHUFFLE_SMALL_DIR, "completed.csv")

# Create necessary directories
os.makedirs(TOOL_SHUFFLE_SMALL_DIR, exist_ok=True)
os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info(f"Ensuring data directories exist: {TOOL_SHUFFLE_SMALL_DIR}, {TRAIN_OUTPUT_DIR}")

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')


try:
    processing_scripts = [
        'data-preprocessing/tool_shuffle_small_processed.py',
    ]
    
    for script in processing_scripts:
        logger.info(f"Executing processing script: {script}")
        with open(script) as file:
            exec(file.read())
    
    logger.info("All data processing steps completed successfully.")
except Exception as e:
    print(f"Error during preprocessing: {e}")
    exit()  

def merge_csv_files(file_paths, output_path):
    try:
        # Load and merge CSV files
        df_list = [pd.read_csv(file) for file in file_paths]
        merged_df = pd.concat(df_list, ignore_index=True)
        merged_df.to_csv(output_path, index=False)
        return merged_df.shape[0]
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

# Merge files and get the number of rows
num_rows = merge_csv_files([MODIFIED_FILE_PATH], MERGED_FILE_PATH)

if num_rows is not None:
    print(f"Merged file saved at {MERGED_FILE_PATH} with {num_rows} rows.")

modified_df = pd.read_csv(MERGED_FILE_PATH)

# Collect all tools from all rows
modified_df = modified_df.sample(frac=1, random_state=572343).reset_index(drop=True)
all_tools = []
for tools_str in modified_df['tools']:
    # print('tools_str', tools_str)
    tools = json.loads(tools_str)
    all_tools.extend(tools)

merged_df = pd.read_csv(MERGED_FILE_PATH)

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

merged_df['tools'] = merged_df['tools'].apply(update_tools)
merged_df.to_csv(COMPLETED_FILE_PATH, index=False)
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

# Login to Hugging Face
login(token=os.getenv("HF_TOKEN"))

# Load and verify dataset
dataset = load_dataset('csv', data_files=COMPLETED_FILE_PATH)['train']

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
            # Add system message with tools
            system_content = """You are an expert in composing functions in python code. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out with the message "{ No Valid Tool Call Provided. }". If the given question lacks the parameters required by the function, also point it out with the message "{ No Valid Tool Call Provided. }".
You should only return the function call in tools call sections.

You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke:"""
            
            system_content += f"\n{json.dumps(tools, indent=2)}"
            formatted_conversation = [{"role": "user", "content": system_content}]
            formatted_conversation[0]['content'] += '\n \n'
            for message in conversation:
                if message.get('role') == 'user':
                    formatted_conversation[0]['content'] += message.get('content', '')

            for message in conversation:
                if message.get('role') == 'assistant':
                    # formatted_conversation.append(message)
                    # Clean up empty function calls
                    content = message.get('content', '')
                    if content == '()':
                        content = ''
                    formatted_conversation.append({"role": "assistant", "content": content})

            # Use the built-in chat template for other models
            print('formatted_conversation', formatted_conversation)
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
# Save the entire dataset as train.csv
train_df = pd.DataFrame(formatted_ds)
train_df.to_csv(os.path.join(TRAIN_OUTPUT_DIR, "train.csv"), index=False)

print(f"Dataset saved successfully to {TRAIN_OUTPUT_DIR}/train.csv!")

