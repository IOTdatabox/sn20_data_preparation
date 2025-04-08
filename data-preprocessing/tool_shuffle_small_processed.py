import json
import pandas as pd
from ast import literal_eval
import logging
import os
from datasets import load_dataset

TOOL_SHUFFLE_SMALL_DIR = "data/tool_shuffle_small"
ORIGIN_FILE_PATH = os.path.join(TOOL_SHUFFLE_SMALL_DIR, "origin.csv")
MODIFIED_FILE_PATH = os.path.join(TOOL_SHUFFLE_SMALL_DIR, "modified.csv")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# fetch from BitAgent/tool_shuffle_small and save it to tool_shuffle_small_origin.csv
def huggingface_loader(dataset_name, split="train", name=None):
    logger.debug(f"Loading {dataset_name}")
    
    logger.debug("Loading from web ...")
    ds = load_dataset(dataset_name, split=split, name=name, token=os.getenv("HF_TOKEN", None))
    # save data as csv
    df = pd.DataFrame(ds)
    df.to_csv(ORIGIN_FILE_PATH, index=False)
    logger.debug(f"Loaded and saved to {ORIGIN_FILE_PATH}")
    return ds

huggingface_loader("BitAgent/tool_shuffle_small")
df = pd.read_csv(ORIGIN_FILE_PATH)

    
def format_arguments(arguments):
    """Formats function arguments into a string suitable for a function call."""
    formatted_args = []
    # print('arguments', arguments)
    for key, value in arguments.items():
        if isinstance(value, str):
            # Escape single quotes and wrap in single quotes
            # formatted_value = f"'{value.replace("'", r"\'")}'"
            formatted_value = "'" + value.replace("'", r"\'") + "'"

        elif isinstance(value, bool):
            # Boolean values should be True or False without quotes
            formatted_value = str(value)
        elif isinstance(value, (int, float)):
            # Numeric values should be without quotes
            formatted_value = str(value)
        elif value is None:
            formatted_value = 'None'
        else:
            # Use JSON serialization for non-string types (lists, dicts, etc.)
            # formatted_value = json.dumps(value)
            formatted_value = value
        formatted_args.append(f"{key}={formatted_value}")
    return ", ".join(formatted_args)


# Process each row in the DataFrame
for index, row in df.iterrows():
    # Parse the conversation string into a list of dictionaries
    try:
        conversation = json.loads(row['conversation'])
        tools = json.loads(row['tools'])
    except json.JSONDecodeError:
        conversation = literal_eval(row['conversation'])

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
          print('error', e)
        

    # Remove all entries where role is 'tool call'
    conversation = [entry for entry in conversation if entry['role'] != 'tool call']

    # modify tools structure according to openAI function calling schema
    # df.at[index, 'tools'] = json.dumps(modify_tools(tools))
    df.at[index, 'tools'] = json.dumps(tools)
    # Convert the conversation back to a JSON string
    df.at[index, 'conversation'] = json.dumps(conversation)

# Save the modified DataFrame
df.to_csv(MODIFIED_FILE_PATH, index=False)