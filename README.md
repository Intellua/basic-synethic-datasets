# JDN Synthetic Dataset Generator

This project generates synthetic question-answer datasets from markdown files using LangChain and OpenAI's API with structured outputs.

## Features

- Processes all `.md` files in the `files/` directory
- Generates exactly 10 questions and answers for each markdown file using Pydantic structured outputs
- Uses OpenAI's GPT models via LangChain with reliable parsing
- Incremental CSV writing - saves progress after each file to prevent data loss
- Outputs results to `./output/generated_questions_and_answers.csv` for easy analysis

## Setup

### 1. Install Dependencies

Make sure you have Python 3.13+ installed, then install the dependencies:

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 2. Set up Environment Variables

Create a `.env` file in the project root and configure your OpenAI settings:

```bash
cp .env.example .env
```

Edit the `.env` file and configure the following variables:

```
# Required: Your OpenAI API key
OPENAI_API_KEY=sk-your-actual-api-key-here

# Optional: OpenAI model to use (defaults to gpt-4o-mini)
OPENAI_MODEL=gpt-4o-mini
```

**Available Models:**
- `gpt-4o-mini` (default) - Fast and cost-effective, good for structured outputs
- `gpt-4o` - More capable but slower and more expensive
- `gpt-3.5-turbo` - Older model, may have less reliable structured outputs

### 3. Add Markdown Files

Place your markdown files (`.md`) in the `files/` directory. The script will process all `.md` files it finds in this directory.

## Usage

Run the script:

```bash
python main.py
```

The script will:
1. Check for your OpenAI API key and model configuration
2. Find all `.md` files in the `files/` directory
3. Create the `./output/` directory and initialize the CSV file
4. Process each file sequentially to generate exactly 10 questions and answers using structured outputs
5. Save results incrementally to `./output/generated_questions_and_answers.csv` after each file (preventing data loss)
6. Display progress and final statistics

## Output Format

The generated CSV file (`./output/generated_questions_and_answers.csv`) contains three columns:
- `source_file`: The path to the original markdown file
- `question`: The generated question
- `answer`: The corresponding answer

The file is written incrementally, so you can monitor progress and won't lose data if the process is interrupted.

## Example

A sample markdown file (`files/sample_article.md`) is included to demonstrate the functionality. After running the script, you'll get a CSV with questions like:

```csv
source_file,question,answer
files/sample_article.md,"What is machine learning?","Machine learning is a subset of artificial intelligence..."
files/sample_article.md,"What are the three main types of machine learning?","The three main types are supervised learning, unsupervised learning, and reinforcement learning."
...
```

## Configuration

### Environment Variables
- **`OPENAI_MODEL`**: Set the OpenAI model to use (required)
- **`OPENAI_API_KEY`**: Your OpenAI API key (required)

### Code Configuration
You can modify the following settings in `main.py`:

- **Temperature**: Adjust creativity by changing the `temperature` parameter (currently 0.7)
- **Number of Questions**: Modify the `QuestionAnswerSet` Pydantic model to generate more or fewer questions (currently exactly 10)
- **Output Directory**: Change the `output_file` path (currently `./output/generated_questions_and_answers.csv`)
- **Prompt Template**: Customize the question generation prompt to focus on specific aspects

### Structured Output Benefits
The script uses Pydantic models with LangChain's structured output feature, which ensures:
- Reliable parsing of exactly 10 Q&A pairs per file
- No parsing errors from malformed responses
- Consistent JSON schema validation
- Better error handling and debugging

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