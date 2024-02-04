import modal
import os
import time
from pydantic import BaseModel

vol = modal.Volume.persisted("srma")
stub = modal.Stub(
    "srma-retool",
    image=modal.Image.debian_slim().pip_install(
        ["google-generativeai", "pandas", "openai"]
    ),
)
WORKSPACE = "jzsang"
MOUNT_DIR = "/data"
EXCLUDED_ABSTRACTS = "review_2821_irrelevant_csv_20240130175158.csv"
INCLUDED_ABSTRACTS = "review_2821_included_csv_20240127132513.csv"
IS_OPENAI = False
ASSETS_PATH = os.path.join(os.path.dirname(__file__), "assets")


def get_url(label):
    return f"https://{WORKSPACE}--{label}-dev.modal.run"


class Item(BaseModel):
    name: str
    number: int = 42
    preprompt: str
    prompt: str
    model: str = "gemini-pro"
    few_shot: int = 0


class Result(BaseModel):
    llm_answer: str
    correct: bool
    skipped: bool
    prompt: str
    error: str = ""


@stub.function()
@modal.web_endpoint(label="submit", method="POST")
def f(item: Item):
    correct_count = 0
    total = 0
    skipped = 0
    results = []
    for res in gen.map(
        range(item.number),
        return_exceptions=True,
        kwargs={"preprompt": item.preprompt, "prompt": item.prompt, "model": item.model, "few_shot": item.few_shot},
    ):
        if isinstance(res, ValueError):
            raise Exception("Parameters set incorrectly! Aborting..")
        if isinstance(res, Exception):
            if len(res.args) > 0:
                error = str(res.args[0])
            else:
                error = "Unknown error"
            results.append(
                Result(
                    llm_answer="No answer",
                    correct=False,
                    skipped=True,
                    prompt="No prompt",
                    error=error,
                )
            )
            skipped += 1
            continue
        total += 1
        correct, prompt, llm_answer = res
        results.append(
            Result(llm_answer=llm_answer, correct=correct, skipped=False, prompt=prompt)
        )
        if correct:
            correct_count += 1

    return {
        "results": results,
        "total_correct": correct_count,
        "total": total,
        "total_skipped": skipped,
    }


@stub.function()
def setup(model):
    if model == "gemini-pro":
        import google.generativeai as genai

        genai.configure(api_key=os.environ["SERVICE_ACCOUNT_JSON"])
        # Set up the model
        generation_config = {
            "temperature": 1.0,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 4096,
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

        model = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
        return lambda prompt: model.generate_content([prompt]).text
    elif model == "gpt-3.5-turbo" or model == "gpt-4-0125-preview" or model == "gpt-3.5-turbo-0125":
        from openai import OpenAI

        client = OpenAI()

        return (
            lambda prompt: client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model,
            )
            .choices[0]
            .message.content
        )
    else:
        raise ValueError("Invalid model")


def is_excluded(text, test_abstract):
    if "4832094" in text and "0918345" in text:
        raise Exception(
            f"Invalid case: LLM misinterpreted the prompt when looking at {test_abstract}"
        )
    elif "4832094" in text:
        return True
    elif "0918345" in text:
        return False
    else:
        raise Exception(
            f"Invalid case: LLM misinterpreted the prompt when looking at {test_abstract}"
        )


def sample_abstract(excluded, seed_generator, n=1):
    import pandas as pd
    import os
    import numpy as np

    file = EXCLUDED_ABSTRACTS if excluded else INCLUDED_ABSTRACTS

    df = pd.read_csv(os.path.join("/data", file), keep_default_na=False)
    abstracts = df["Abstract"].replace("", np.nan).dropna().sample(n=n, random_state=seed_generator).tolist()

    return abstracts


@stub.function(
    volumes={MOUNT_DIR: vol},
    secrets=[
        modal.Secret.from_name("srma-openai"),
        modal.Secret.from_name("srma-gemini"),
    ],
)
def gen(seed, preprompt=None, prompt=None, model=None, few_shot=0):
    if preprompt is None or prompt is None:
        raise ValueError("Missing parameters")
    import random
    import numpy as np
    seed_generator = np.random.default_rng(seed)

    # Whether to pick from the excluded or included pile
    random.seed(seed)
    excluded = random.choice([True, False])

    # Get a random abstract
    test_abstracts = sample_abstract(excluded, seed, n=1)
    test_abstract = test_abstracts[0]

    few_shot_text = ""
    if few_shot:
        few_shot_text += "\nExamples:\n"
        number_include = few_shot // 2
        number_exclude = few_shot - number_include

        few_shot_exclusionary_abstracts = sample_abstract(True, seed_generator, n=number_exclude)
        few_shot_inclusionary_abstracts = sample_abstract(False, seed_generator, n=number_include)

        for abstract in few_shot_exclusionary_abstracts:
            few_shot_text += f"""
---
{abstract}

This article should be excluded.
"""
        for abstract in few_shot_inclusionary_abstracts:
            few_shot_text += f"""
---
{abstract}

This article should be included.

"""
        few_shot_text += "---\nEND OF EXAMPLES\n"

    # Setup the model
    model = setup.local(model)

    # Create the prompt and get the response
    prompt = f"{preprompt}\n{few_shot_text}\n{test_abstract}\n{prompt}\n"
    llm_answer = None
    llm_answer = model(prompt)

    # Check if the response is correct
    predicted_value = is_excluded(llm_answer, test_abstract)
    return predicted_value == excluded, prompt, llm_answer


@stub.local_entrypoint()
def main():
    correct_count = 0
    total = 0
    skipped = 0
    for res in gen.map(range(2), return_exceptions=True):
        if isinstance(res, Exception):
            skipped += 1
            print(res)
            continue
        total += 1
        correct, prompt, response = res
        if correct:
            correct_count += 1
        else:
            print(f"Failed response: {response}")

    print(f"Total correct: {correct_count}/{total}, Total skipped: {skipped}")
