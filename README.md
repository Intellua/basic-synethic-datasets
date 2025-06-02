# JDN Synthetic Dataset Generator & Model Evaluation System

This project generates synthetic question-answer datasets from markdown files and provides a comprehensive evaluation system to test multiple AI models against the generated data using MLflow.

## Features

### Dataset Generation
- Processes all `.md` files in the `files/` directory
- Generates exactly 10 questions and answers for each markdown file using Pydantic structured outputs
- Uses OpenAI's GPT models via LangChain with reliable parsing
- Incremental CSV writing - saves progress after each file to prevent data loss
- Outputs results to `./output/generated_questions_and_answers.csv` for easy analysis

### Model Evaluation
- Evaluates multiple OpenAI models (GPT-4, GPT-4-turbo, GPT-3.5-turbo, GPT-4o-mini) against generated Q&A data
- Uses MLflow for experiment tracking and model comparison
- Generates detailed evaluation reports in CSV format
- Provides aggregated metrics and per-question analysis
- Supports custom system prompts and model configurations

## Setup

### 1. Install Dependencies

Make sure you have Python 3.13+ installed, then install the dependencies:

```bash
# Using uv (recommended)
uv sync
uv pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

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

### 1. Generate Dataset

Run the dataset generation script:

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

### 2. Evaluate Models

After generating the dataset, you can evaluate multiple models against it:

```bash
uvx mlflow ui --host 127.0.0.1 --port 8111
uvx run run_evaluation.py
```

Or use the evaluation system directly:

```bash
uvx run eval.py
```

The evaluation system will:
1. Load the generated Q&A dataset from `./output/generated_questions_and_answers.csv`
2. Test multiple OpenAI models (GPT-4, GPT-4-turbo, GPT-3.5-turbo, GPT-4o-mini)
3. Use MLflow to track experiments and log metrics
4. Generate detailed evaluation reports in `./output/evaluations/`
5. Create a summary report comparing all models
6. Display aggregated results and metrics

### 3. View Results

To view detailed evaluation results in MLflow UI:

```bash
mlflow ui
```

Then navigate to http://localhost:5000 to explore:
- Model performance metrics
- Individual question-answer evaluations
- Experiment comparisons
- Detailed logs and artifacts

## Output Format

### Dataset Generation Output

The generated CSV file (`./output/generated_questions_and_answers.csv`) contains three columns:
- `source_file`: The path to the original markdown file
- `question`: The generated question
- `answer`: The corresponding answer

The file is written incrementally, so you can monitor progress and won't lose data if the process is interrupted.

### Model Evaluation Output

The evaluation system creates several output files in `./output/evaluations/`:

1. **Individual Model Results**: `{model_name}_evaluation_results.csv`
   - Contains detailed evaluation metrics for each question
   - Includes model responses, ground truth comparisons, and scoring metrics
   - Useful for analyzing model performance on specific questions

2. **Summary Report**: `evaluation_summary.csv`
   - Aggregated metrics across all evaluated models
   - Comparison table with key performance indicators
   - Overview of which models performed best

3. **MLflow Artifacts**:
   - Experiment tracking data
   - Model artifacts and metadata
   - Detailed logs and metrics history
   - Accessible via MLflow UI at http://localhost:5000

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
- **`OPENAI_API_KEY`**: Your OpenAI API key (required)
- **`OPENAI_MODEL`**: Set the OpenAI model to use for dataset generation (optional, defaults to gpt-4o-mini)

### Dataset Generation Configuration
You can modify the following settings in `main.py`:

- **Temperature**: Adjust creativity by changing the `temperature` parameter (currently 1)
- **Number of Questions**: Modify the `QuestionAnswerSet` Pydantic model to generate more or fewer questions (currently exactly 10)
- **Output Directory**: Change the `output_file` path (currently `./output/generated_questions_and_answers.csv`)
- **Prompt Template**: Customize the question generation prompt to focus on specific aspects

### Model Evaluation Configuration
You can customize the evaluation system in `eval.py`:

- **Models to Evaluate**: Modify the `get_model_configurations()` method to add/remove models
- **System Prompts**: Customize the system prompts for each model
- **Evaluation Metrics**: MLflow provides built-in question-answering metrics
- **Output Directory**: Results are saved to `./output/evaluations/`

#### Available Evaluation Metrics
MLflow's question-answering evaluator provides several metrics:
- **Exact Match**: Binary score for exact string matches
- **Token Count**: Number of tokens in model responses
- **Latency**: Response time for each question
- **Additional metrics**: Depending on MLflow version and configuration

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