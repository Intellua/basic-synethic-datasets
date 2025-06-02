#!/usr/bin/env python3
"""
Simple script to run the model evaluation system.

This script demonstrates how to use the evaluation system to test multiple
OpenAI models against the generated question-answer dataset.
"""

from eval import ModelEvaluator
import sys
import os

def main():
    """Run the evaluation system."""
    print("🔬 MLflow Model Evaluation Runner")
    print("="*50)
    
    # Check if the CSV file exists
    csv_path = "./output/generated_questions_and_answers.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found!")
        print("Please run main.py first to generate the question-answer dataset.")
        return 1
    
    # Check environment
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key in a .env file or as an environment variable.")
        return 1
    
    try:
        # Initialize and run evaluations
        evaluator = ModelEvaluator(csv_path)
        results = evaluator.run_all_evaluations()
        
        print(f"\n🎉 Successfully evaluated {len(results)} models!")
        print("\n🔍 To view detailed results:")
        print("1. Run: uvx mlflow ui --host 127.0.0.1 --port 8111")
        print("2. Open: http://localhost:8111")
        print("3. Check the 'output/evaluations/' directory for CSV files")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())