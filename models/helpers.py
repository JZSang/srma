from constants import MAX_COMPLETION_TOKENS, TOKENIZER
from models.messages import generate_messages
from models.openai import chat_completions_create_params, extract_from_response_openai

def setup_model(model, model_seed):
    if model == "gemini-pro":
        import google.generativeai as genai
        import os

        genai.configure(api_key=os.environ["SERVICE_ACCOUNT_JSON"])
        # Set up the model
        generation_config = {
            "temperature": 1.0,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": MAX_COMPLETION_TOKENS,
        }

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
        ]

        # TODO: add seed
        client = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

        async def returned_model(prompt):
            return (
                await client.generate_content_async(generate_messages(model, prompt))
            ).text

        return returned_model
    elif (
        model == "gpt-3.5-turbo"
        or model == "gpt-4-0125-preview"
        or model == "gpt-3.5-turbo-0125"
        or model == "gpt-4-turbo-2024-04-09"
    ):
        from openai import AsyncOpenAI

        client = AsyncOpenAI()

        async def returned_model(prompt):
            return (
                extract_from_response_openai(
                    await client.chat.completions.create(**chat_completions_create_params(model, prompt, model_seed)),
                    is_json=False
                )
            )

        return returned_model
    elif model == "open-mixtral-8x22b-2404" or model == "mistral-large-2402":
        from mistralai.async_client import MistralAsyncClient

        client = MistralAsyncClient()

        async def returned_model(prompt):
            return (
                (
                    await client.chat(
                        model=model,
                        messages=generate_messages(model, prompt),
                        random_seed=model_seed,
                        max_tokens=MAX_COMPLETION_TOKENS,
                    )
                )
                .choices[0]
                .message.content
            )

        return returned_model
    else:
        raise ValueError("Invalid model setup model")

def calculate_tokens(model, prompt, max_token_response):
    if model == "gemini-pro":
        # gemini doesn't limit by tokens
        return 1
    elif (
        model == "gpt-3.5-turbo"
        or model == "gpt-4-0125-preview"
        or model == "gpt-3.5-turbo-0125"
        or model == "gpt-4-turbo-2024-04-09"
    ):
        import tiktoken

        tokens_per_message = 3
        tokens_per_name = 1
        encoding = tiktoken.get_encoding(TOKENIZER)
        num_tokens = 0
        for message in generate_messages(model, prompt):
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        num_tokens += max_token_response
        return num_tokens
    elif model == "open-mixtral-8x22b-2404" or model == "mistral-large-2402":
        from mistral_common.protocol.instruct.messages import (
            UserMessage,
        )
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        from mistral_common.tokens.instruct.normalize import ChatCompletionRequest

        tokenizer_v3 = MistralTokenizer.v3()

        tokenized = tokenizer_v3.encode_chat_completion(
            ChatCompletionRequest(
                messages=[UserMessage(content=prompt)],
                model=model,
            )
        )
        num_tokens = len(tokenized.tokens)
        num_tokens += max_token_response
        return num_tokens
    else:
        raise ValueError("Invalid model calculate tokens")