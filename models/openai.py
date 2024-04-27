from constants import MAX_COMPLETION_TOKENS
from openai.types.chat import CompletionCreateParams
from models.messages import generate_messages


def chat_completions_create_params(model: str, prompt: str, model_seed: int) -> CompletionCreateParams:
    return {
        "messages": generate_messages(model, prompt),
        "model": model,
        "max_tokens": MAX_COMPLETION_TOKENS,
        "seed": model_seed,
    }

def extract_from_response_openai(response, is_json=False):
    if is_json:
        return response["choices"][0]["message"]["content"]
    return response.choices[0].message.content