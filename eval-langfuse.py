from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
import requests
import os
import uuid
import json
from mlflow.metrics.genai import EvaluationExample, faithfulness
import datetime
from dotenv import load_dotenv
import random

load_dotenv()

# init
langfuse = Langfuse()

models = [
    "llama3.3:70b",
    "gemma3:4b",
    "qwen3:14b"
]

temperatures = [
    0.6,
    1.0,
    0.1
]

prompts = [
    langfuse.get_prompt("jdn-prompt"),
    langfuse.get_prompt("jdn-helpfulness"),
]

cartesian_product = []
for model in models:
    for temperature in temperatures:
        for prompt in prompts:
            cartesian_product.append((model, temperature, prompt))
random.shuffle(cartesian_product)

print("Cartesian product of models and temperatures:")
for model, temperature, prompt in cartesian_product:
    print(f"Model: {model}, Temperature: {temperature}, Prompt: {prompt.name}")

def helpfulness_llm_as_a_judge(query: str, generation: str, ground_truth: str):
    body = {
      "model": os.getenv("OPENAI_EVAL_MODEL", "qwen3:30b-a3b"),
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"Identify the helpfulness of a response by analyzing the relationship between the query, answer, and expected answer. \n\nStart with reasoning and end with the response. Consider references to external elements and other aspects when determining helpfulness.\n\n# Steps\n\n1. **Understand the input components**: Clearly differentiate between the query, the given answer, and the expected answer.\n2. **Analyze the Answer**: \n   - Check if the answer directly addresses the query.\n   - Evaluate the accuracy and relevance of the response in relation to the expected answer.\n3. **Consider External Elements**: \n   - Assess if the answer appropriately refers to external elements, providing added value or clarification.\n4. **Formulate Reasoning**: \n   - Base your reasoning on the alignment between the answer and the expected answer, while noting any helpful references to external information.\n5. **Determine Helpfulness**: \n   - Conclude how helpful the answer is based on the analysis, supported by your reasoning.\n\n# Output Format\n\nThe output should be structured into two parts:\n- **Reasoning**: A detailed explanation of the analysis.\n- **Response**: A conclusion stating the helpfulness of the answer.\n\n# Examples\n\n**Example 1:**\n\n- **Query**: \"What is the capital of France?\"\n- **Answer**: \"The capital of France is Paris.\"\n- **Expected Answer**: \"Paris.\"\n\n**Reasoning**: The given answer correctly identifies the capital of France as Paris, which matches the expected answer. No additional references are used, but the information is accurate and directly addresses the query.\n\n**Response**: The answer is helpful.\n\n**Example 2:**\n\n- **Query**: \"What are some health benefits of eating apples?\"\n- **Answer**: \"Apples are good for your heart, may help prevent cancer, and boost your immunity. They are also linked to a lower risk of diabetes.\"\n- **Expected Answer**: \"Apples help in improving heart health and boosting immunity.\"\n\n**Reasoning**: The answer provides an expanded list of health benefits compared to the expected answer, which adds value. It correctly includes the benefits mentioned in the expected answer and introduces relevant additional information, increasing helpfulness by referencing other significant health benefits.\n\n**Response**: The answer is very helpful.\n\n# Notes\n\n- Ensure reasoning is coherent and fully supports the response.\n- Consider each answer\"s context and specificity to the query when evaluating helpfulness.\n- External references should enhance the response\"s value or clarity.\\n\\nInput:\\nQuery:\\n```\\n{query}\\n```\\n\\nGeneration:\\n```\\n{generation}\\n```\\n\\nGround truth:\\n```\\n{ground_truth}\\n```\\n\\n\\n\\n"
            }
          ]
        }
      ],
      "response_format": {
        "type": "json_schema",
        "json_schema": {
          "name": "helpfulness_detection",
          "strict": True,
          "schema": {
            "type": "object",
            "properties": {
              "reasoning": {
                "type": "string",
                "description": "The reasoning behind the helpfulness score."
              },
              "score": {
                "type": "number",
                "description": "A float value representing the helpfulness score between 0 and 1."
              }
            },
            "required": [
              "reasoning",
              "score"
            ],
            "additionalProperties": True
          }
        }
      },
      "temperature": 0.6,
      "max_completion_tokens": 2048,
      "top_p": 0.95,
      "frequency_penalty": 0,
      "presence_penalty": 0
    }

    api_endpoint = os.getenv("OPENAI_OLLAMA_URL", "http://lxc-ai01:3000/ollama/v1")  + "/chat/completions"
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
            stream=False,
            timeout=120
        )
        
        response.raise_for_status()
        
        # Handle non-streaming response
        result = response.json()["choices"][0]["message"]["content"]
        print("Raw result:", result)

        result_json = json.loads(result)
        print("Parsed result JSON:", result_json)
        if "score" in result_json:
            return result_json["reasoning"], float(result_json["score"])
        else:
            print("Score not found in response:", result_json)
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        print("Response content:", response.text if 'response' in locals() else "No response")
        return None

def score_llm_as_a_judge(query: str, generation: str, ground_truth: str):
    body = {
      "model": os.getenv("OPENAI_EVAL_MODEL", "qwen3:30b-a3b"),
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"Evaluate the correctness of the generation on a continuous scale from 0 to 1. A generation can be considered correct (Score: 1) if it includes all the key facts from the ground truth and if every fact presented in the generation is factually supported by the ground truth or common sense.\\n\\nExample:\\nQuery: Can eating carrots improve your vision?\\nGeneration: Yes, eating carrots significantly improves your vision, especially at night. This is why people who eat lots of carrots never need glasses. Anyone who tells you otherwise is probably trying to sell you expensive eyewear or does not want you to benefit from this simple, natural remedy. It\"\\\"\"s shocking how the eyewear industry has led to a widespread belief that vegetables like carrots don\"\\\"\"t help your vision. People are so gullible to fall for these money-making schemes.\\nGround truth: Well, yes and no. Carrots won\"\\\"\"t improve your visual acuity if you have less than perfect vision. A diet of carrots won\"\\\"\"t give a blind person 20/20 vision. But, the vitamins found in the vegetable can help promote overall eye health. Carrots contain beta-carotene, a substance that the body converts to vitamin A, an important nutrient for eye health. An extreme lack of vitamin A can cause blindness. Vitamin A can prevent the formation of cataracts and macular degeneration, the world\"\\\"\"s leading cause of blindness. However, if your vision problems aren\"\\\"\"t related to vitamin A, your vision won\"\\\"\"t change no matter how many carrots you eat.\\nScore: 0.1\\nReasoning: While the generation mentions that carrots can improve vision, it fails to outline the reason for this phenomenon and the circumstances under which this is the case. The rest of the response contains misinformation and exaggerations regarding the benefits of eating carrots for vision improvement. It deviates significantly from the more accurate and nuanced explanation provided in the ground truth.\\n\\n\\n\\nInput:\\nQuery:\\n```\\n{query}\\n```\\n\\nGeneration:\\n```\\n{generation}\\n```\\n\\nGround truth:\\n```\\n{ground_truth}\\n```\\n\\n\\n\\nThink step by step."
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
                "reasoning": {
                "type": "string",
                "description": "The reasoning behind the score assigned, explaining how the actual answer compares to the expected answer."
              },
              "score": {
                "type": "number",
                "description": "The score assigned based on the quality of the actual answer compared to the expected answer."
              }
            },
            "required": [
              "score",
              "reasoning"
            ],
            "additionalProperties": False
          }
        }
      },
      "temperature": 0.6,
      "max_completion_tokens": 2048,
      "top_p": 0.95,
      "frequency_penalty": 0,
      "presence_penalty": 0
    }

    api_endpoint = os.getenv("OPENAI_OLLAMA_URL", "http://lxc-ai01:3000/ollama/v1")  + "/chat/completions"
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
            stream=False,
            timeout=120
        )
        
        response.raise_for_status()
        
        # Handle non-streaming response
        result = response.json()["choices"][0]["message"]["content"]
        print("Raw result:", result)

        result_json = json.loads(result)
        print("Parsed result JSON:", result_json)
        if "score" in result_json:
            return result_json["reasoning"], float(result_json["score"])
        else:
            print("Score not found in response:", result_json)
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        print("Response content:", response.text if 'response' in locals() else "No response")
        return None

@observe(capture_input=False)
def eval_llm_as_a_judge(
    user_content,
    system_content=None,
    stream=False,
    model="qwen3:14b",
    chat_id=None,
    temperature=0.6,
    files=None,
    variables=None,
    api_endpoint=None,
    item = None,
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
    
    # Set default variables if not provided
    if variables is None:
        variables = {
            "USER": "Langfuse"
        }
    
    # Construct the request payload
    payload = {
        "stream": stream,
        "model": model,
        "messages": messages,
        "temperature": temperature,

        # OpenWebUI specific parameters
        "variables": variables,
        "chat_id": chat_id,
        "files": files,
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
        
        start = datetime.datetime.now()
        response = requests.post(
            api_endpoint,
            json=payload,
            headers=headers,
            stream=stream,
            timeout=120
        )
        end = datetime.datetime.now()
        
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code} from API")
            print(f"Response content: {response.json()}")
            return None
        
        # Handle non-streaming response
        json = response.json()
        output = json["choices"][0]["message"]["content"]

        llm_as_a_judge_reason, llm_as_a_judge_score = score_llm_as_a_judge(item.input, output, item.expected_output)
        print(f"Score for {user_content}: {llm_as_a_judge_score}")

        llm_as_a_judge_helpfulreason, llm_as_a_judge_helpfulscore = score_llm_as_a_judge(item.input, output, item.expected_output)
        print(f"Helpfulness score for {user_content}: {llm_as_a_judge_helpfulscore}")

        langfuse_context.score_current_observation(
          name="score",
          value=llm_as_a_judge_score,
          data_type="NUMERIC",  # optional, inferred if not provided
          comment=llm_as_a_judge_reason
        )
        langfuse_context.score_current_observation(
          name="helpfulness_score",
          value=llm_as_a_judge_helpfulscore,
          data_type="NUMERIC",  # optional, inferred if not provided
          comment=llm_as_a_judge_helpfulreason
        )
        langfuse_context.score_current_observation(
          name="duration",
          value=(end - start).total_seconds(),
          data_type="NUMERIC",  # optional, inferred if not provided
          comment="Score from LLM as a judge for correctness",
        )
        langfuse_context.update_current_observation(model=model, start_time=start, end_time=end, usage_details={
            "input": json["usage"]["prompt_tokens"],
            "output": json["usage"]["completion_tokens"]
        }, input=user_content, output=output)

        return output
            
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None

# @observe(capture_input=False, capture_output=False, as_type="generation")
def run_experiment(experiment_name: str, system_prompt: str, model: str, temperature: float):
  dataset = langfuse.get_dataset("wiki_questions")

  for index, item in enumerate(dataset.items):
    if index > 4:
      print(f"Skipping item {index} as it exceeds the limit of 50 items.")
      break
    # item.observe() returns a trace_id that can be used to add custom evaluations later
    # it also automatically links the trace to the experiment run
    with item.observe(run_name=experiment_name,
                      run_description=f"Model: {model}, Temperature: {temperature}", run_metadata={
        "model": model,
        "temperature": temperature,
        "experiment_name": experiment_name
    }) as trace_id:
      print(f"Running evaluation for: {item.input} with trace ID: {trace_id}")
 
      # run application, pass input and system prompt
      _ = eval_llm_as_a_judge(item.input, system_prompt,
                                     item=item,
                                     model=model,
                                     temperature=temperature,
                                     )

      langfuse_context.flush()
      langfuse.flush()
  langfuse_context.update_current_observation(input=experiment_name, model=model)

if __name__ == "__main__":
    
    for (model, temperature, prompt) in cartesian_product:
        experiment_name = f"jdn_wiki-{model}-{prompt.name}-{temperature}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        print(f"Running experiment with model: {model}")
        run_experiment(experiment_name, prompt.prompt, model, temperature)
        langfuse_context.flush()
        langfuse.flush()
    
    print(f"Running experiment: {experiment_name}")
    langfuse_context.flush()
    langfuse.flush()