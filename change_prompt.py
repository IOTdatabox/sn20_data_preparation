import pandas as pd
import re

# Define the original and updated system prompt
original_prompt = r"You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.\nIf none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.\nYou should only return the function call in tools call sections.\n\nYou SHOULD NOT include any other text in the response."
updated_prompt = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function call in tools call sections.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1="params_string_value1", params_name2=params_value2...), func_name2(params)]
Notice that any values that are strings must be put in quotes like this: "params_string_value1"
Boolean values must always start with an uppercase letter** (`True`, `False`).
You SHOULD NOT include any other text in the response. If NO function is appropriate, return an **EMPTY string** (`""`)"""

# Load CSV file (update 'your_file.csv' with actual filename)
csv_filename = 'data/reference/train.csv'
df = pd.read_csv(csv_filename)

# Check if the original prompt exists in the DataFrame
if df['text'].str.contains(original_prompt, regex=True).any():
    print("Original prompt found in the DataFrame.")
else:
    print("Original prompt NOT found in the DataFrame.")

# Replace occurrences of the original prompt
df['text'] = df['text'].replace(to_replace=original_prompt, value=updated_prompt, regex=True)

# Save the updated CSV (update 'updated_file.csv' as needed)
updated_csv_filename = 'data/reference/updated_train.csv'
df.to_csv(updated_csv_filename, index=False)

print(f"Updated CSV saved as {updated_csv_filename}")