import modal
from pydantic import BaseModel

TOKENIZER = "cl100k_base"

### MODAL INITIALIZERS
stub = modal.Stub(
    "srma-retool-tokenizer",
    image=modal.Image.debian_slim().pip_install(
        ["google-generativeai", "pandas", "openai", "tiktoken"]
    ),
)

class Tokenizer(BaseModel):
    result: dict
    model: str

@stub.function()
@modal.web_endpoint(label="tokenize", method="POST")
def calculate_tokens(params: Tokenizer):
    model = params.model
    
    if model == "gemini-pro" or model == "gemini-1.5-flash-latest":
        # gemini doesn't limit by tokens
        return {**params.result}
    elif (
        model == "gpt-3.5-turbo"
        or model == "gpt-4-0125-preview"
        or model == "gpt-3.5-turbo-0125"
    ):
        import tiktoken
        encoding = tiktoken.get_encoding(TOKENIZER)
        
        ret = {"llm_answer": len(encoding.encode(params.result["llm_answer"])), "prompt": len(encoding.encode(params.result["prompt"]))}

        return {**params.result, "token_counts": ret}
    elif (
        model == "open-mixtral-8x22b-2404"
        or model == "mistral-large-2402"
    ):
        raise ValueError("Should not be needed")
    else:
        raise ValueError("Invalid model calculate tokens")
