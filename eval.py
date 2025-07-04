import mlflow
import openai
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()

mlflow.set_tracking_uri('http://localhost:8111')

system_prompt = """
Provide concise and professional responses to internal queries, adhering to the guidelines for Jan De Nul Group's JdnGPT assistant.
**Background Information:**
JdnGPT is designed as an internal conversational assistant exclusively for Jan De Nul Group employees. It serves to augment daily work by leveraging organizational knowledge, established procedures, and industry best practices. JdnGPT is finely tuned to support the unique demands of a large, innovative maritime and civil engineering company, covering activities ranging from dredging and offshore construction to environmental remediation and project management.
**Core Functions and Scope:**
- Assist with questions regarding Jan De Nul’s services, operations, project guidelines, internal documentation, and workflows.
- Support information retrieval, report drafting, data analysis, technical clarifications, and procedural compliance.
- Foster safety, sustainability, transparency, and professional excellence in all responses.
- Provide guidance aligning with internal policies, confidentiality requirements, and safety regulations. Refer users to appropriate internal channels when necessary.
**Knowledge Base:**
- Trained on Jan De Nul public materials, internal best practices, and sector-specific terminology as of October 2023.
- No access to real-time company data or project specifics unless provided during interactions.
**Ethics and Privacy:**
- Maintain confidentiality and do not disclose sensitive information.
- Avoid advice outside of the assistant’s scope. Clarify boundaries and encourage contact with experts when needed.
**Interaction Style:**
- Use a professional, respectful, and approachable tone.
- Adjust explanations for varying technical expertise levels.
- Encourage safe practices and continuous improvement in alignment with Jan De Nul’s values.
# Steps
1. Assess the user’s query and determine its alignment with the core functions and scope.
2. Retrieve and provide information within JdnGPT’s knowledge base.
3. If required, guide users to appropriate internal resources or personnel for additional assistance.
4. Ensure the tone is professional and explanations are tailored to the user’s expertise level.
# Output Format
- Responses should be concise and professional.
- Start all yes/no questions with 'Yes' or 'No'.
- Include any necessary clarifications or recommendations following the initial answer.
# Examples
**Example 1:**
**Input:** Is it safe to proceed with the dredging operation as planned?
**Output:**
No, for safety confirmation, please consult the latest safety report and adhere to internal guidelines.
**Example 2:**
**Input:** Can you summarize the procedure for offshore installation?
**Output:**
Yes, offshore installation follows these main steps: [List Steps]. For detailed procedures, refer to the internal documentation available in the [specific system].

# Important
- If no documents are available, inform the user that JdnGPT cannot access real-time data or specific project details.
- When providing answers, ALWAYS mention the source file from which the information was derived!
"""

class ModelEvaluator:
    """Evaluation system for testing multiple models against generated Q&A data."""
    
    def __init__(self, csv_path: str = "./output/generated_questions_and_answers.csv"):
        self.csv_path = csv_path
        self.output_dir = Path("./output/evaluations")
        self.output_dir.mkdir(exist_ok=True)
        
        # Check if OpenAI API key is set
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY environment variable not set")
    
    def load_eval_data(self) -> pd.DataFrame:
        """Load and transform the CSV data into MLflow evaluation format."""
        print(f"Loading evaluation data from {self.csv_path}")
        
        # Read the CSV generated by main.py
        df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(df)} question-answer pairs")
        
        # Transform to MLflow evaluation format
        eval_data = pd.DataFrame({
            "inputs": df["question"].tolist(),
            "ground_truth": df["answer"].tolist(),
            "source_file": df["source_file"].tolist()
        })
        
        return eval_data
    
    def get_model_configurations(self) -> List[Dict[str, Any]]:
        """Define the models to evaluate."""
        return [
            {
                "name": "qwen3:14b",
                "model": "qwen3:14b",
                "system_prompt": system_prompt + " /no_think"
            },
            {
                "name": "gemma3:4b",
                "model": "gemma3:4b",
                "system_prompt": system_prompt
            },
            {
                "name": "llama3.3:70b",
                "model": "llama3.3:70b",
                "system_prompt": system_prompt
            }

            # {
            #     "name": "gpt-4.1-nano",
            #     "model": "gpt-4.1-nano",
            #     "system_prompt": system_prompt
            # },
            # {
            #     "name": "gpt-4.1-mini",
            #     "model": "gpt-4.1-mini",
            #     "system_prompt": system_prompt
            # }
        ]
    
    def evaluate_model(self, model_config: Dict[str, Any], eval_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate a single model using MLflow."""
        model_name = model_config["name"]
        model = model_config["model"]
        system_prompt = model_config["system_prompt"]
        
        print(f"\nEvaluating model: {model_name}")
        
        with mlflow.start_run(run_name=f"eval_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            # Log model parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model", model)
            mlflow.log_param("system_prompt", system_prompt)
            mlflow.log_param("eval_data_size", len(eval_data))
            
            try:
                # Wrap the model as an MLflow model
                logged_model_info = mlflow.openai.log_model(
                    model=model,
                    task=openai.chat.completions,
                    artifact_path="model",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "{question}"},
                    ],
                    files=[
                        {
                            "id": "5124e1a2-9908-4b7a-8945-6fb63f2cea6e",
                            "name": "Wiki",
                            "type": "collection",
                            "status": "processed"
                        }
                    ]
                )
                
                # Use predefined question-answering metrics to evaluate our model
                results = mlflow.evaluate(
                    logged_model_info.model_uri,
                    eval_data,
                    targets="ground_truth",
                    model_type="question-answering",
                    evaluator_config={
                        "col_mapping": {
                            "inputs": "inputs",
                            "targets": "ground_truth"
                        }
                    },
                    extra_metrics=[mlflow.metrics.latency(), mlflow.metrics.toxicity()]
                )
                
                print(f"✅ Evaluation completed for {model_name}")
                print(f"Aggregated metrics: {results.metrics}")
                
                # Save detailed results to CSV
                eval_table = results.tables["eval_results_table"]
                csv_filename = f"{model_name}_evaluation_results.csv"
                csv_path = self.output_dir / csv_filename
                eval_table.to_csv(csv_path, index=False)
                print(f"💾 Detailed results saved to: {csv_path}")
                
                # Log metrics to MLflow
                for metric_name, metric_value in results.metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value)
                
                return {
                    "model_name": model_name,
                    "metrics": results.metrics,
                    "csv_path": str(csv_path),
                    "run_id": run.info.run_id
                }
                
            except Exception as e:
                print(f"❌ Error evaluating {model_name}: {str(e)}")
                mlflow.log_param("error", str(e))
                return {
                    "model_name": model_name,
                    "error": str(e),
                    "run_id": run.info.run_id
                }
    
    def run_all_evaluations(self) -> List[Dict[str, Any]]:
        """Run evaluations for all configured models."""
        print("🚀 Starting model evaluations...")
        
        # Load evaluation data
        eval_data = self.load_eval_data()
        
        # Get model configurations
        model_configs = self.get_model_configurations()
        
        print(f"📊 Will evaluate {len(model_configs)} models on {len(eval_data)} questions")
        
        # Run evaluations
        results = []
        for i, model_config in enumerate(model_configs, 1):
            print(f"\n{'='*50}")
            print(f"Model {i}/{len(model_configs)}: {model_config['name']}")
            print(f"{'='*50}")
            
            result = self.evaluate_model(model_config, eval_data)
            results.append(result)
        
        # Create summary report
        self.create_summary_report(results)
        
        return results
    
    def create_summary_report(self, results: List[Dict[str, Any]]):
        """Create a summary report of all evaluations."""
        print("\n📋 Creating summary report...")
        
        summary_data = []
        for result in results:
            if "error" not in result:
                row = {"model_name": result["model_name"]}
                row.update(result["metrics"])
                row["csv_path"] = result["csv_path"]
                row["run_id"] = result["run_id"]
                summary_data.append(row)
            else:
                summary_data.append({
                    "model_name": result["model_name"],
                    "error": result["error"],
                    "run_id": result["run_id"]
                })
        
        # Save summary to CSV
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / "evaluation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"📄 Summary report saved to: {summary_path}")
        print("\n🎯 Evaluation Summary:")
        print(summary_df.to_string(index=False))
        
        return summary_path


def main():
    """Main function to run the evaluation system."""
    print("🔬 MLflow Model Evaluation System")
    print("="*50)
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        # Run all evaluations
        results = evaluator.run_all_evaluations()
        
        print(f"\n🎉 Evaluation completed!")
        print(f"📊 Evaluated {len(results)} models")
        print(f"📁 Results saved in: {evaluator.output_dir}")
        
        # Print MLflow tracking info
        print(f"\n🔍 View detailed results in MLflow UI:")
        print("Run: mlflow ui")
        print("Then navigate to http://localhost:5000")
        
    except Exception as e:
        print(f"❌ Error running evaluations: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())