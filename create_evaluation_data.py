import os
import logging
import json
import random
import pandas as pd
from datasets import load_dataset, load_from_disk

EVALUATION_OUTPUT_DIR = "data/evaluation"
TOOL_SHUFFLE_SMALL_DIR = "data/tool_shuffle_small"
ORIGIN_FILE_PATH = os.path.join(TOOL_SHUFFLE_SMALL_DIR, "origin.csv")
EVALUATION_FILE_PATH = os.path.join(EVALUATION_OUTPUT_DIR, "evaluation.csv")

os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info(f"Ensuring data directories exist: {TOOL_SHUFFLE_SMALL_DIR}, {EVALUATION_OUTPUT_DIR}")

def add_extra_tool_calls():
   # ADD EXTRAB TOOLS IN TOOLS COLUMNS
    modified_df = pd.read_csv(ORIGIN_FILE_PATH)

    # Collect all tools from all rows
    all_tools = []
    for tools_str in modified_df['tools']:
        # print('tools_str', tools_str)
        tools = json.loads(tools_str)
        all_tools.extend(tools)

    errored_rows = 0
    # Update tools column with JSON-safe formatting
    def update_tools(tools_str):
        try:
            global errored_rows
            selected_tool = random.choice(all_tools)
            tools_list = json.loads(tools_str)
            for _ in range(3):
                selected_tool = random.choice(all_tools)
                tools_list.append(selected_tool)
                # Randomly choose to append or prepend each tool
                # if random.choice([True, False]):
                #     tools_list.append(selected_tool)
                # else:
                #     tools_list.insert(0, selected_tool)
            return json.dumps(tools_list)
        except Exception as e:
                errored_rows += 1
                print(f"Unexpected error: {e}", tools_str)
                return tools_str
  
    modified_df['tools'] = modified_df['tools'].apply(update_tools)
    modified_df.to_csv(EVALUATION_FILE_PATH, index=False)
    print(f"Successfully added tool name to evaluation.csv and these are errored rows", errored_rows)

def create_bitagent_shuffle_dataset():
    try:
        add_extra_tool_calls()
    except Exception as e:
        logger.error(f"Error processing BitAgent dataset: {str(e)}")

if __name__ == "__main__":
    create_bitagent_shuffle_dataset()