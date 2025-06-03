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

def score_llm_as_a_judge(query: str, generation: str, ground_truth: str):
    body = {
      "model": "qwen3:14b",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"Evaluate the correctness of the generation on a continuous scale from 0 to 1. A generation can be considered correct (Score: 1) if it includes all the key facts from the ground truth and if every fact presented in the generation is factually supported by the ground truth or common sense.\\n\\nExample:\\nQuery: Can eating carrots improve your vision?\\nGeneration: Yes, eating carrots significantly improves your vision, especially at night. This is why people who eat lots of carrots never need glasses. Anyone who tells you otherwise is probably trying to sell you expensive eyewear or does not want you to benefit from this simple, natural remedy. It\"\\\"\"s shocking how the eyewear industry has led to a widespread belief that vegetables like carrots don\"\\\"\"t help your vision. People are so gullible to fall for these money-making schemes.\\nGround truth: Well, yes and no. Carrots won\"\\\"\"t improve your visual acuity if you have less than perfect vision. A diet of carrots won\"\\\"\"t give a blind person 20/20 vision. But, the vitamins found in the vegetable can help promote overall eye health. Carrots contain beta-carotene, a substance that the body converts to vitamin A, an important nutrient for eye health. An extreme lack of vitamin A can cause blindness. Vitamin A can prevent the formation of cataracts and macular degeneration, the world\"\\\"\"s leading cause of blindness. However, if your vision problems aren\"\\\"\"t related to vitamin A, your vision won\"\\\"\"t change no matter how many carrots you eat.\\nScore: 0.1\\nReasoning: While the generation mentions that carrots can improve vision, it fails to outline the reason for this phenomenon and the circumstances under which this is the case. The rest of the response contains misinformation and exaggerations regarding the benefits of eating carrots for vision improvement. It deviates significantly from the more accurate and nuanced explanation provided in the ground truth.\\n\\n\\n\\nInput:\\nQuery: {query}\\nGeneration: {generation}\\nGround truth: {ground_truth}\\n\\n\\nThink step by step."
            }
          ]
        }
      ],
      "response_format": {
        "type": "json_schema",
        "json_schema": {
          "name": "llm_score_system",
          "strict": True,
          "schema": {
            "type": "object",
            "properties": {
              "score": {
                "type": "number",
                "description": "The score assigned based on the quality of the actual answer compared to the expected answer."
              }
            },
            "required": [
              "score"
            ],
            "additionalProperties": False
          }
        }
      },
      "temperature": 0.6,
      "max_completion_tokens": 2048,
      "top_p": 0.95,
      "min_p": 0,
      "top_k": 20,
      "frequency_penalty": 0,
      "presence_penalty": 0
    }

    api_endpoint = os.getenv("OPENAI_OLLAMA_URL", "http://lxc-ai01:3000/ollama")  + "/v1/chat/completions"
    try:
        # Make the API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"
        }
        
        response = requests.post(
            api_endpoint,
            json=body,
            headers=headers,
            stream=False
        )
        
        response.raise_for_status()
        
        # Handle non-streaming response
        result = response.json()["choices"][0]["message"]["content"]

        result_json = json.loads(result)
        if "score" in result_json:
            return float(result_json["score"])
        else:
            print("Score not found in response:", result_json)
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None

@observe()
def run_my_custom_llm_app(
    user_content,
    system_content=None,
    stream=False,
    model="qwen3:14b",
    chat_id=None,
    max_tokens=5000,
    temperature=0.6,
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
            "num_ctx": num_ctx,
        },
        "messages": messages,
        "files": files,
        "variables": variables,
        "top_p": 0.95,
        "min_p": 0,
        "top_k": 20,
    }
    
    # Get API endpoint from environment or use provided one
    if api_endpoint is None:
        api_endpoint = os.getenv("OPENAI_BASE_URL", "http://lxc-ai01:3000/api") + "/chat/completions"
    
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
  dataset = langfuse.get_dataset("wiki_questions")

  for item in dataset.items:
    # item.observe() returns a trace_id that can be used to add custom evaluations later
    # it also automatically links the trace to the experiment run
    with item.observe(run_name=experiment_name) as trace_id:
 
      # run application, pass input and system prompt
      output = run_my_custom_llm_app(item.input, system_prompt)

      llm_as_a_judge_score = score_llm_as_a_judge(item.input, output, item.expected_output)
    
      # optional: add custom evaluation results to the experiment trace
      # we use the previously created example evaluation function
      langfuse.score(
        id="llm_as_a_judge_score", # optional, can be used as an indempotency key to update the score subsequently
        trace_id=trace_id,
        name="llm_as_a_judge_score",
        value=llm_as_a_judge_score,
        data_type="NUMERIC", # optional, inferred if not provided
        comment="Factual correctness",
      )
      langfuse.score(
        id="output", # optional, can be used as an indempotency key to update the score subsequently
        trace_id=trace_id,
        name="Output",
        value=output,
        comment="Output from the LLM",
      )

if __name__ == "__main__":
    # Run example usage
    run_experiment("Jan De Nul Group - JdnGPT", system_prompt.prompt)
    langfuse.flush()