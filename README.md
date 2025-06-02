# JDN Synthetic Dataset Generator

This project generates synthetic question-answer datasets from markdown files using LangChain and OpenAI's API.

## Features

- Processes all `.md` files in the `files/` directory
- Generates 10 questions and answers for each markdown file
- Uses OpenAI's GPT models via LangChain
- Outputs results to a CSV file for easy analysis

## Setup

### 1. Install Dependencies

Make sure you have Python 3.13+ installed, then install the dependencies:

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 2. Set up OpenAI API Key

Create a `.env` file in the project root and add your OpenAI API key:

```bash
cp .env.example .env
```

Edit the `.env` file and replace `your_openai_api_key_here` with your actual OpenAI API key:

```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 3. Add Markdown Files

Place your markdown files (`.md`) in the `files/` directory. The script will process all `.md` files it finds in this directory.

## Usage

Run the script:

```bash
python main.py
```

The script will:
1. Check for your OpenAI API key
2. Find all `.md` files in the `files/` directory
3. Process each file to generate 10 questions and answers
4. Save the results to `generated_questions_and_answers.csv`

## Output Format

The generated CSV file contains three columns:
- `source_file`: The path to the original markdown file
- `question`: The generated question
- `answer`: The corresponding answer

## Example

A sample markdown file (`files/sample_article.md`) is included to demonstrate the functionality. After running the script, you'll get a CSV with questions like:

```csv
source_file,question,answer
files/sample_article.md,"What is machine learning?","Machine learning is a subset of artificial intelligence..."
files/sample_article.md,"What are the three main types of machine learning?","The three main types are supervised learning, unsupervised learning, and reinforcement learning."
...
```

## Configuration

You can modify the following settings in `main.py`:

- **Model**: Change the OpenAI model by modifying the `model` parameter in `ChatOpenAI()`
- **Temperature**: Adjust creativity by changing the `temperature` parameter
- **Number of Questions**: Modify the prompt template to generate more or fewer questions

## Requirements

- Python 3.13+
- OpenAI API key
- Dependencies listed in `pyproject.toml`

## Error Handling

The script includes error handling for:
- Missing OpenAI API key
- Empty or missing markdown files
- API errors during question generation
- File reading/writing errors

## License

This project is open source and available under the MIT License.