"""
Configuration settings for the data preparation project.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model configuration
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
DATASET_LENGTH = 3600
MAX_SEQ_LENGTH = 1024

# Directory structure
TOOL_SHUFFLE_SMALL_DIR = "data/tool_shuffle_small"
TRAIN_OUTPUT_DIR = "data/train"
EVALUATION_OUTPUT_DIR = "data/evaluation"

# File paths
ORIGIN_FILE_PATH = os.path.join(TOOL_SHUFFLE_SMALL_DIR, "origin.csv")
MODIFIED_FILE_PATH = os.path.join(TOOL_SHUFFLE_SMALL_DIR, "modified.csv")
MERGED_FILE_PATH = os.path.join(TOOL_SHUFFLE_SMALL_DIR, "merged.csv")
COMPLETED_FILE_PATH = os.path.join(TOOL_SHUFFLE_SMALL_DIR, "completed.csv")
EVALUATION_FILE_PATH = os.path.join(EVALUATION_OUTPUT_DIR, "evaluation.csv")
TRAIN_FILE_PATH = os.path.join(TRAIN_OUTPUT_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(TRAIN_OUTPUT_DIR, "test.csv")

# Hugging Face configuration
HF_TOKEN = os.getenv('HF_TOKEN')
HF_USERNAME = os.getenv('HF_USERNAME')

# System prompt for tokenization
# SYSTEM_PROMPT = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.

# ### **Rules for Function Calls:**

# 1. **Strict Output Format:**
#    - **ONLY return the function call in the tool call section.**
#    - If NO function is appropriate, return an **EMPTY string** (`""`).
#    - Always return responses in the format of [func_name1(params_name1="params_string_value1", params_name2=params_value2...), func_name2(params)]
#    - **DO NOT respond with "None", "No functions available", or any text-based explanations.**
#    - **DO NOT return JSON, code blocks, or any other format.**
#    - **Strictly follow these rules in ALL cases.**

# 2. **Boolean Formatting:**
#    - **Boolean values must always start with an uppercase letter** (`True`, `False`).
#    - **Strictly follow this rule in ALL cases.**

# 3. **Handling Arguments:**
#    - **Include all required arguments** as defined in the function specification.
#    - **DO NOT use underscores (`_`) in dictionary keys or argument values.**
#      - ✅ Correct: `category="science fiction"`
#      - ❌ Incorrect: `category="science_fiction"`  
#    - **Avoid file extensions (`.jpg`, `.png`) in image-related arguments.**
#      - ✅ Correct: `hero_image="exoplanets"`
#      - ❌ Incorrect: `hero_image="exoplanet_illustration.jpg"`  
#    - **Avoid overly specific or creative values in expected fields.**
#      - ✅ Correct: `thumbnail_url="wind turbine"`
#      - ❌ Incorrect: `thumbnail_url="https://example.com/wind_turbine.jpg"`

# Here is a list of functions in JSON format that you can invoke:"""

SYSTEM_PROMPT = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function call in tools call sections.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1="params_string_value1", params_name2=params_value2...), func_name2(params)]
Notice that any values that are strings must be put in quotes like this: "params_string_value1"
You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke."""
