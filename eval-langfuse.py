from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
import requests
import os
import uuid
import json
from mlflow.metrics.genai import EvaluationExample, faithfulness

from dotenv import load_dotenv

load_dotenv()

# init
langfuse = Langfuse()
system_prompt = langfuse.get_prompt("jdn-prompt")

@observe()
def run_my_custom_llm_app(
    user_content,
    system_content=None,
    stream=False,
    model="qwen3:14b",
    chat_id=None,
    max_tokens=5000,
    temperature=1,
    num_ctx=32000,
    files=None,
    variables=None,
    api_endpoint=None
):
    """
    Make a custom API call with configurable parameters
    
    Args:
        user_content (str): The user's message content
        system_content (str): The system prompt content
        stream (bool): Whether to stream the response
        model (str): The model to use
        chat_id (str): Chat ID, generates a UUID if not provided
        max_tokens (int): Maximum tokens for response
        temperature (float): Temperature for response generation
        num_ctx (int): Context size
        files (list): List of file objects with id, name, type, status
        variables (dict): Variables to pass to the API
        api_endpoint (str): The API endpoint URL
    """
    
    # Generate chat_id if not provided
    if chat_id is None:
        chat_id = str(uuid.uuid4())
    
    # Set default system content if not provided
    if system_content is None:
        system_content = "You are a helpfull precise respectfull assistant. You answer questions brief but comprehensive. Do not speculate or make up information./no_think"
    
    # Construct messages
    messages = [
        {
            "content": system_content,
            "role": "system"
        },
        {
            "content": user_content,
            "role": "user"
        }
    ]
    
    # Set default files if not provided
    if files is None:
        files = [
            {
                "id": "5124e1a2-9908-4b7a-8945-6fb63f2cea6e",
                "name": "Wiki",
                "type": "collection",
                "status": "processed"
            }
        ]
    
    # Set default variables if not provided
    if variables is None:
        variables = {
            "USER": "Langfuse"
        }
    
    # Construct the request payload
    payload = {
        "stream": stream,
        "model": model,
        "chat_id": chat_id,
        "params": {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "num_ctx": num_ctx
        },
        "messages": messages,
        "files": files,
        "variables": variables
    }
    
    # Get API endpoint from environment or use provided one
    if api_endpoint is None:
        api_endpoint = os.getenv("OPENAI_BASE_URL", "http://lxc-ai01:3000/api/chat/completions")
    
    try:
        # Make the API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"
        }
        
        response = requests.post(
            api_endpoint,
            json=payload,
            headers=headers,
            stream=stream
        )
        
        response.raise_for_status()
        
        # Handle non-streaming response
        return response.json()["choices"][0]["message"]["content"]
            
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None

def run_experiment(experiment_name, system_prompt):
  dataset = langfuse.get_dataset("Wiki Questions")

  for item in dataset.items:
    # item.observe() returns a trace_id that can be used to add custom evaluations later
    # it also automatically links the trace to the experiment run
    with item.observe(run_name=experiment_name) as trace_id:
 
      # run application, pass input and system prompt
      output = run_my_custom_llm_app(item.input, system_prompt)
 
      # optional: add custom evaluation results to the experiment trace
      # we use the previously created example evaluation function
      langfuse.score(
        id="unique_id", # optional, can be used as an indempotency key to update the score subsequently
        trace_id=trace_id,
        name="correctness",
        value=0.9,
        data_type="NUMERIC", # optional, inferred if not provided
        comment="Factually correct",
    )

if __name__ == "__main__":
    # Run example usage
    run_experiment("Jan De Nul Group - JdnGPT", system_prompt.prompt)
    langfuse.flush()