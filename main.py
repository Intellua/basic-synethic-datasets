import os
import csv
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

# Load environment variables from .env file
load_dotenv()

class QuestionAnswer(BaseModel):
    """A single question-answer pair."""
    question: str = Field(description="A question based on the content")
    answer: str = Field(description="The corresponding answer to the question")

class QuestionAnswerSet(BaseModel):
    """A set of 10 question-answer pairs."""
    qa_pairs: List[QuestionAnswer] = Field(
        description="Exactly 10 question-answer pairs based on the content",
        min_length=10,
        max_length=10
    )

def read_markdown_file(file_path):
    """Read content from a markdown file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def generate_questions_and_answers(content, llm):
    """Generate 10 questions and answers based on the given content using structured output."""
    # Create a structured output LLM
    structured_llm = llm.with_structured_output(QuestionAnswerSet)
    
    prompt_template = PromptTemplate(
        input_variables=["content"],
        template="""
Based on the following content, generate exactly 10 diverse questions and their corresponding answers.
The questions should cover different aspects of the content including:
- Key concepts and definitions
- Examples and applications
- Comparisons and contrasts
- Benefits and limitations
- How-to or procedural questions

Make sure each answer is comprehensive and accurate based on the provided content. Each question and answer should be in flemish.

Content:
{content}
"""
    )
    
    prompt = prompt_template.format(content=content)
    response = structured_llm.invoke([HumanMessage(content=prompt)])
    
    return response.qa_pairs

def write_header_to_csv(output_file):
    """Write CSV header."""
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['source_file', 'question', 'answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

def append_to_csv(qa_data, output_file):
    """Append Q&A data to CSV file and flush immediately."""
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['source_file', 'question', 'answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        for row in qa_data:
            writer.writerow(row)
        csvfile.flush()  # Ensure data is written to disk immediately

def main():
    print("Starting jdn-synthetic-dataset question generation...")
    
    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key in a .env file or as an environment variable.")
        return
    
    # Initialize OpenAI LLM
    llm = ChatOpenAI(
        model=os.getenv('OPENAI_MODEL'),  # Using gpt-4o-mini for better structured output support
        temperature=1
    )
    
    # Find all .md files in the files directory
    files_dir = Path("files")
    md_files = list(files_dir.glob("*.md"))
    
    if not md_files:
        print("No .md files found in the 'files' directory.")
        print("Please add some .md files to the 'files' directory and try again.")
        return
    
    print(f"Found {len(md_files)} .md file(s) to process:")
    for file in md_files:
        print(f"  - {file}")
    
    # Prepare output file
    output_file = "./output/generated_questions_and_answers.csv"
    
    # Create output directory if it doesn't exist
    Path("./output").mkdir(exist_ok=True)
    
    # Write CSV header
    write_header_to_csv(output_file)
    print(f"Created output file: {output_file}")
    
    total_qa_count = 0
    
    # Process each markdown file
    for i, md_file in enumerate(md_files, 1):
        print(f"\nProcessing file {i}/{len(md_files)}: {md_file}")
        
        try:
            # Read file content
            content = read_markdown_file(md_file)
            
            if not content.strip():
                print(f"  Warning: {md_file} is empty, skipping...")
                continue
            
            print(f"  Content length: {len(content)} characters")
            
            # Generate Q&A pairs using structured output
            print("  Generating questions and answers...")
            qa_pairs = generate_questions_and_answers(content, llm)
            
            print(f"  Generated {len(qa_pairs)} Q&A pairs")
            
            # Convert to dict format and append to CSV immediately
            file_qa_data = []
            for qa in qa_pairs:
                file_qa_data.append({
                    'source_file': str(md_file),
                    'question': qa.question,
                    'answer': qa.answer
                })
            
            # Append to CSV and flush
            append_to_csv(file_qa_data, output_file)
            total_qa_count += len(file_qa_data)
            
            print(f"  ✅ Saved {len(file_qa_data)} Q&A pairs to CSV")
        
        except Exception as e:
            print(f"  ❌ Error processing {md_file}: {str(e)}")
            continue
    
    if total_qa_count > 0:
        print(f"\n🎉 Successfully generated {total_qa_count} Q&A pairs from {len(md_files)} file(s)!")
        print(f"Results saved to: {output_file}")
    else:
        print("\n❌ No Q&A pairs were generated.")

if __name__ == "__main__":
    main()
