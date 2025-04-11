# SN20 Data Preparation

This project prepares datasets for training and evaluation of language models with tool-calling capabilities.

## Project Structure

```
.
├── config.py                  # Configuration settings
├── utils.py                   # Utility functions
├── create_train_data.py       # Script to create training data
├── create_evaluation_data.py  # Script to create evaluation data
├── upload_data.py             # Script to upload data to Hugging Face Hub
├── data-preprocessing/        # Data preprocessing scripts
│   └── tool_shuffle_small_processed.py
├── data/                      # Data directories
│   ├── tool_shuffle_small/    # Small tool shuffle dataset
│   ├── train/                 # Training data
│   └── evaluation/            # Evaluation data
└── requirements.txt           # Project dependencies
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with the following variables:
   ```
   HF_TOKEN=your_huggingface_token
   HF_USERNAME=your_huggingface_username
   ```

## Usage

### Creating Training Data

To create training data, run:

```
python create_train_data.py
```

This will:
1. Download the tool_shuffle_small dataset from Hugging Face
2. Process the dataset to format tool calls
3. Add random tools to create a more challenging training set
4. Tokenize the dataset using the Qwen2-7B-Instruct model
5. Save the tokenized dataset to `data/train/train.csv`

### Creating Evaluation Data

To create evaluation data, run:

```
python create_evaluation_data.py
```

This will:
1. Read the original dataset
2. Add extra tool calls to create a more challenging evaluation set
3. Save the modified dataset to `data/evaluation/evaluation.csv`

### Uploading Data to Hugging Face Hub

To upload the processed dataset to Hugging Face Hub, run:

```
python upload_data.py
```

This will:
1. Load the dataset from a CSV file
2. Tokenize the dataset
3. Filter out empty strings
4. Split the dataset into train and test sets
5. Push the dataset to Hugging Face Hub

## Dependencies

- pandas
- datasets
- python-dotenv
- transformers
- huggingface-hub
- jinja2 