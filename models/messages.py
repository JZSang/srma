def generate_messages(model, prompt):
    if model == "gemini-pro" or model == "gemini-1.5-flash-latest":
        return [prompt]
    elif (
        model == "gpt-3.5-turbo"
        or model == "gpt-4-0125-preview"
        or model == "gpt-3.5-turbo-0125"
        or model == "gpt-4-turbo-2024-04-09"
        or model == "gpt-4o-2024-05-13"
        or model == "gpt-4o-2024-08-06"
        or model == "gpt-4o-2024-11-20"
        or model == "o1-2024-12-17"
        or model == "o3-mini-2025-01-31"
    ):
        return [{"role": "user", "content": prompt}]
    elif model in ["claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219"]:
        return [{"role": "user", "content": prompt}]
    elif model == "open-mixtral-8x22b-2404" or model == "mistral-large-2402":
        from mistralai.models.chat_completion import ChatMessage

        return [ChatMessage(role="user", content=prompt)]
    else:
        raise ValueError(f"Invalid model generate_messages {model}")
