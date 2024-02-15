import modal
import os
import time
from pydantic import BaseModel
from dataclasses import dataclass, field
import asyncio
import numpy as np

vol = modal.Volume.persisted("srma")
stub = modal.Stub(
    "srma-retool",
    image=modal.Image.debian_slim().pip_install(
        ["google-generativeai", "pandas", "openai", "tiktoken"]
    ),
)
stub.result_queue = modal.Queue.new()
stub.status_tracker = modal.Dict.new()
WORKSPACE = "jzsang"
MOUNT_DIR = "/data"
EXCLUDED_ABSTRACTS = "review_2821_irrelevant_csv_20240130175158.csv"
INCLUDED_ABSTRACTS = "review_2821_included_csv_20240127132513.csv"
IS_OPENAI = False
ASSETS_PATH = os.path.join(os.path.dirname(__file__), "assets")
TOKENIZER = "cl100k_base"
MAX_COMPLETION_TOKENS = 1536

USAGE_LIMITS = {
    "gpt-3.5-turbo": {
        "max_requests_per_minute": 500,
        "max_tokens_per_minute": 60000,
    },
    "gpt-4-0125-preview": {
        "max_requests_per_minute": 500,
        "max_tokens_per_minute": 150000,
    },
    "gpt-3.5-turbo-0125": {
        "max_requests_per_minute": 500,
        "max_tokens_per_minute": 60000,
    },
    "gemini-pro": {
        "max_requests_per_minute": 60,
        "max_tokens_per_minute": float("inf"),
    },
}


def get_url(label):
    return f"https://{WORKSPACE}--{label}-dev.modal.run"


class Item(BaseModel):
    name: str
    number: int = 1
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
    predicted_value: str = ""
    is_excluded_value: str = ""
    actual_value: str = ""


@stub.function()
@modal.web_endpoint(label="submit", method="POST")
def f(item: Item):
    correct_count = 0
    total_actually_processed = 0
    skipped = 0
    results = []

    total_prompts = item.number
    # until we get like tier 6 rate limiting, we should really only use one machine
    prompts_per_cpu = 99999999
    number_of_cpus = ((total_prompts - 1) // prompts_per_cpu) + 1

    # clear the persisted queue
    stub.result_queue.get_many(stub.result_queue.len(), False)
    assert stub.result_queue.len() == 0

    # function: now split the total prompts into the number of cpus with remainders
    def split_prompts(total_prompts, number_of_cpus):
        prompts_per_cpu = total_prompts // number_of_cpus
        prompts = [prompts_per_cpu] * number_of_cpus
        for i in range(total_prompts % number_of_cpus):
            prompts[i] += 1
        return [[i, prompt] for i, prompt in enumerate(prompts)]

    for res in gen.starmap(
        split_prompts(total_prompts, number_of_cpus),
        kwargs={
            "preprompt": item.preprompt,
            "prompt": item.prompt,
            "model": item.model,
            "few_shot": item.few_shot,
        },
    ):
        pass
    # Currently, failures not thrown during text generation completely stop execution.
    if stub.result_queue.len() != total_prompts:
        return {"results": [], "total_correct": -1, "total": -1, "total_skipped": -1}
    for res in stub.result_queue.get_many(total_prompts):
        if res[0]:
            results.append(
                Result(
                    llm_answer="No answer",
                    correct=False,
                    skipped=True,
                    prompt=res[2],
                    error=str(res[1]),
                )
            )
            skipped += 1
            continue
        total_actually_processed += 1
        _, correct, prompt, llm_answer, predicted_value, actual_value = res
        results.append(
            Result(
                llm_answer=llm_answer,
                correct=correct,
                skipped=False,
                prompt=prompt,
                predicted_value=predicted_value,
                is_excluded_value=actual_value,
                actual_value=actual_value,
            )
        )
        if correct:
            correct_count += 1

    status_tracker: StatusTracker = stub.status_tracker["object"]
    
    return {
        "results": results,
        "total_correct": correct_count,
        "total": total_actually_processed,
        "total_skipped": skipped,
        "status_tracker": status_tracker.__dict__,
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
    ):
        from openai import AsyncOpenAI

        client = AsyncOpenAI()

        async def returned_model(prompt):
            return (
                (
                    await client.chat.completions.create(
                        messages=generate_messages(model, prompt),
                        model=model,
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
    else:
        raise ValueError("Invalid model calculate tokens")


def is_excluded(text, test_abstract):
    # search area to the very end, maybe about 100 characters
    text = text[-100:]
    if "XXX" in text and "YYY" in text:
        raise Exception(
            f"Invalid case: Impossible to decide LLM's answer: {text} {test_abstract}"
        )
    elif "XXX" in text:
        return "excluded"
    elif "YYY" in text:
        return "included"
    else:
        raise Exception(
            f"Invalid case: LLM did not print an answer: {text} {test_abstract}"
        )


def generate_messages(model, prompt):
    if model == "gemini-pro":
        return [prompt]
    elif (
        model == "gpt-3.5-turbo"
        or model == "gpt-4-0125-preview"
        or model == "gpt-3.5-turbo-0125"
    ):
        return [{"role": "user", "content": prompt}]
    else:
        raise ValueError(f"Invalid model generate_messages {model}")

def load_abstracts():
    import pandas as pd
    excluded_file = EXCLUDED_ABSTRACTS
    included_file = INCLUDED_ABSTRACTS
    # Load up volume files in memory
    df_excluded = pd.read_csv(os.path.join("/data", excluded_file), keep_default_na=False)
    abstracts_excluded = (
        df_excluded["Abstract"]
        .replace("", np.nan)
        .dropna()
    )
    df_included = pd.read_csv(os.path.join("/data", included_file), keep_default_na=False)
    abstracts_included = (
        df_included["Abstract"]
        .replace("", np.nan)
        .dropna()
    )
    return {
        "excluded": abstracts_excluded,
        "included": abstracts_included
    }


@stub.function(
    volumes={MOUNT_DIR: vol},
    secrets=[
        modal.Secret.from_name("srma-openai"),
        modal.Secret.from_name("srma-gemini"),
    ],
    memory=1024,
    cpu=2.0
)
async def gen(
    seed, num_of_prompts, preprompt=None, prompt=None, model=None, few_shot=0
):
    if preprompt is None or prompt is None or model is None:
        raise ValueError("Missing parameters")
    import asyncio
    import numpy as np
    import logging

    seed_generator = np.random.default_rng(seed)
    # initialize logging
    logging_level = logging.INFO
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        0.001  # 1 ms limits max throughput to 1,000 requests per second
    )
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()
    status_tracker = StatusTracker()
    next_request = None  # variable to hold the next request to call
    # initialize available capacity counts
    max_requests_per_minute = USAGE_LIMITS[model]["max_requests_per_minute"]
    max_tokens_per_minute = USAGE_LIMITS[model]["max_tokens_per_minute"]
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    abstract_files = load_abstracts()

    while True:
        if next_request is None:
            if num_of_prompts:
                logging.info(f"num_of_prompts left: {num_of_prompts}")
                next_request = Abstract(
                    task_id=next(task_id_generator),
                    preprompt=preprompt,
                    prompt=prompt,
                    model=model,
                    few_shot=few_shot,
                    attempts_left=3,
                    excluded=seed_generator.choice([True, False]),
                    abstract_files=abstract_files,
                )
                next_request.setup(seed_generator)
                num_of_prompts -= 1
                status_tracker.num_tasks_started += 1
                status_tracker.num_tasks_in_progress += 1
                logging.info(f"Instantiating task {next_request.task_id}")
            elif not queue_of_requests_to_retry.empty():
                next_request = queue_of_requests_to_retry.get_nowait()
                logging.info(f"Instantiating task {next_request.task_id}")

        current_time = time.time()
        seconds_since_update = current_time - last_update_time
        available_request_capacity = min(
            available_request_capacity
            + max_requests_per_minute * seconds_since_update / 60.0,
            max_requests_per_minute,
        )
        available_token_capacity = min(
            available_token_capacity
            + max_tokens_per_minute * seconds_since_update / 60.0,
            max_tokens_per_minute,
        )
        last_update_time = current_time

        if next_request:
            next_request_tokens = next_request.token_consumption
            if (available_request_capacity >= 1) and (
                available_token_capacity >= next_request_tokens
            ):
                logging.info(
                    f"started: {status_tracker.num_tasks_started}, in-progress: {status_tracker.num_tasks_in_progress}, succeeded: {status_tracker.num_tasks_succeeded}, failed: {status_tracker.num_tasks_failed}, queue length: {queue_of_requests_to_retry.qsize()}, task_id: {next_request.task_id}"
                )
                available_request_capacity -= 1
                available_token_capacity -= next_request_tokens
                next_request.attempts_left -= 1
                asyncio.create_task(
                    next_request.single_abstract(
                        queue_of_requests_to_retry, status_tracker
                    )
                )
                logging.info(f"Task {next_request.task_id} started")
                next_request = None
        if status_tracker.num_tasks_in_progress <= 0:
            break

        # main loop sleeps briefly so concurrent tasks can run
        await asyncio.sleep(seconds_to_sleep_each_loop)

        # if a rate limit error was hit recently, pause to cool down
        seconds_since_rate_limit_error = (
            time.time() - status_tracker.time_of_last_rate_limit_error
        )
        if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
            remaining_seconds_to_pause = (
                seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error
            )
            await asyncio.sleep(remaining_seconds_to_pause)
            # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
            logging.warn(
                f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
            )
    if status_tracker.num_tasks_failed > 0:
        logging.warning(
            f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors should be returned."
        )
    if status_tracker.num_rate_limit_errors > 0:
        logging.warning(
            f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
        )

    stub.status_tracker["object"] = status_tracker


    return 1

### PARALLEL PROCESSING HELPERS ##########################################

@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class Abstract:
    task_id: int

    preprompt: str
    prompt: str
    model: str
    few_shot: int
    excluded: bool
    abstract_files: dict

    attempts_left: int = 0
    result: list = field(default_factory=list)

    final_prompt: str = None
    token_consumption: int = 0

    def sample_abstract(self, excluded, seed_generator, n=1):
        file = "excluded" if excluded else "included"

        abstracts = self.abstract_files[file].sample(n=n, random_state=seed_generator).tolist()

        return abstracts

    def setup(self, seed_generator: np.random.Generator):
        if self.final_prompt:
            return
        # Get a random abstract
        test_abstracts = self.sample_abstract(self.excluded, seed_generator, n=1)
        test_abstract = test_abstracts[0]

        few_shot_text = ""
        if self.few_shot:
            few_shot_text += "\nExamples:\n"
            number_include = self.few_shot // 2
            number_exclude = self.few_shot - number_include

            few_shot_exclusionary_abstracts = self.sample_abstract(
                True, seed_generator, n=number_exclude
            )
            few_shot_inclusionary_abstracts = self.sample_abstract(
                False, seed_generator, n=number_include
            )

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

        # Create the prompt and get the response
        self.final_prompt = f"{self.preprompt}\n{few_shot_text}\nHere is the abstract we are trying to understand:\n{test_abstract}\n\n{self.prompt}\n"
        self.token_consumption = calculate_tokens(
            self.model, self.final_prompt, MAX_COMPLETION_TOKENS
        )

    async def single_abstract(
        self, retry_queue: asyncio.Queue, status_tracker: StatusTracker
    ):
        if self.attempts_left < 0:
            raise ValueError("attempts_left should never be negative")
        error = None
        try:
            ###### BUSINESS
            # Setup the model
            model_async = setup.local(self.model)

            llm_answer = None
            llm_answer = await model_async(self.final_prompt)

            # Check if the response is correct
            predicted_value = is_excluded(llm_answer, self.final_prompt)
            actual_value = "excluded" if self.excluded else "included"
            ###### BUSINESS
        except Exception as e:
            if "Rate limit" in str(e) or "429" in str(e):
                status_tracker.time_of_last_rate_limit_error = time.time()
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                stub.result_queue.put(
                    [True, [str(e) for e in self.result], self.final_prompt]
                )
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            stub.result_queue.put(
                [
                    False,
                    predicted_value == actual_value,
                    self.final_prompt,
                    llm_answer,
                    predicted_value,
                    actual_value,
                ]
            )
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1




##### GENERAL HELPERS ######################################################

def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1