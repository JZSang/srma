import modal
import os
import time
from pydantic import BaseModel
from dataclasses import dataclass, field
from typing import Union
import asyncio
import numpy as np

# Constant texts
FINAL_PROMPT = "{preprompt}\n{few_shot_text}\n# Abstract in investigation: \n{test_abstract}\n\n{prompt}\n"


### LLM SETTINGS
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
TOKENIZER = "cl100k_base"
TEXT_SEED = 1337

### MODAL SETTINGS
WORKSPACE = "jzsang"

### Raw dataset storage
vol = modal.Volume.persisted("srma")
MOUNT_DIR = "/data"
EXCLUDED_ABSTRACTS = "review_2821_irrelevant_csv_20240130175158.csv"
INCLUDED_ABSTRACTS = "review_2821_included_csv_20240127132513.csv"

### GPTCoT storage
MOUNT_DIR_COT = "/cot"
cot_vol = modal.Volume.persisted("srma-cot")


### MODAL INITIALIZERS
stub = modal.Stub(
    "srma-retool",
    image=modal.Image.debian_slim().pip_install(
        ["google-generativeai", "pandas", "openai", "tiktoken"]
    ),
)

### Producer-Consumer Queues
stub.result_queue = modal.Queue.new()
stub.cot_result_queue = modal.Queue.new()

### Process status tracker
stub.status_tracker = modal.Dict.new()


@stub.function()
@modal.web_endpoint(label="status", method="GET")
def get_status(mode):
    return stub.status_tracker[mode].__dict__

@stub.function()
@modal.web_endpoint(label="kill", method="POST")
def kill(mode):
    if mode != "test" and mode != "gptcot":
        raise ValueError(f"Invalid mode {mode}")
    stub.status_tracker[mode + "kill"] = True
    return True

class GenerationTask:
    excluded_abstracts = None
    included_abstracts = None
    gpt_cot = None

    def __init__(
        self,
        mode,
        num_excluded_abstracts,
        num_included_abstracts,
        preprompt,
        prompt,
        model,
        include_dataset,
        exclude_dataset,
        gpt_cot_id=None,
        few_shot_exclude=0,
        few_shot_include=0,
    ):
        self.mode = mode
        if mode == "test":
            self.result_queue = stub.result_queue
            print(
                f"Running test mode with {num_excluded_abstracts} excluded abstracts and {num_included_abstracts} included abstracts"
            )
        elif mode == "gptcot":
            self.result_queue = stub.cot_result_queue
            print(
                f"Running gptcot mode with {num_excluded_abstracts} excluded abstracts and {num_included_abstracts} included abstracts"
            )
            if gpt_cot_id is not None:
                raise ValueError("gpt_cot_id should not be provided for gptcot mode")
            if few_shot_exclude or few_shot_include:
                raise ValueError(
                    "few_shot_exclude or few_shot_include should not be provided for gptcot mode"
                )
        else:
            raise ValueError(f"Invalid mode {mode}")
        # Clear the persisted queue
        self.result_queue.get_many(stub.result_queue.len(), False)
        assert self.result_queue.len() == 0

        self.num_excluded_abstracts = num_excluded_abstracts
        self.num_included_abstracts = num_included_abstracts

        self.preprompt = preprompt
        self.prompt = prompt
        self.model = model

        self.include_dataset = include_dataset
        self.exclude_dataset = exclude_dataset

        self.gpt_cot_id = gpt_cot_id
        self.few_shot_exclude = few_shot_exclude
        self.few_shot_include = few_shot_include
        (
            print(f"Using {self.gpt_cot_id} as gpt_cot_id")
            if self.gpt_cot_id
            else print("No gpt_cot_id provided")
        )
        (
            print(
                f"Using {few_shot_exclude} excluded fewshots and {few_shot_include} included fewshots"
            )
            if (few_shot_exclude or few_shot_include)
            else print("No fewshots provided")
        )
        if self.gpt_cot_id and not (self.few_shot_exclude or self.few_shot_include):
            raise ValueError("gpt_cot_id provided but no fewshots")
        self.total = self.count()

    def count(self):
        return self.num_excluded_abstracts + self.num_included_abstracts

    def is_generated_few_shot(self):
        return bool(self.gpt_cot_id)

    def needs_append_few_shots(self):
        return self.few_shot_exclude or self.few_shot_include

    def load_abstracts(self):
        import pandas as pd
        import os
        import numpy as np

        # Load up volume files in memory
        df_excluded = pd.read_csv(
            os.path.join("/data", self.exclude_dataset), keep_default_na=False
        )
        excluded_abstracts = df_excluded["Abstract"].replace("", np.nan).dropna()
        df_included = pd.read_csv(
            os.path.join("/data", self.include_dataset), keep_default_na=False
        )
        included_abstracts = df_included["Abstract"].replace("", np.nan).dropna()
        self.excluded_abstracts = excluded_abstracts
        self.included_abstracts = included_abstracts

    def get_abstract_series(self, actual_value="excluded"):
        exclude = True if actual_value == "excluded" else False
        if exclude:
            return self.excluded_abstracts
        else:
            return self.included_abstracts

    def get_cot_abstract_dataframe(self, actual_value="excluded"):
        if self.gpt_cot is None:
            raise ValueError("gpt_cot is not loaded")
        return self.gpt_cot[self.gpt_cot["actual_value"] == actual_value]

    def load_gpt_cot(self):
        import pandas as pd
        import os

        if self.gpt_cot_id:
            self.gpt_cot = pd.read_csv(
                os.path.join(MOUNT_DIR_COT, f"cot_{self.gpt_cot_id}.csv")
            )

    def consume(self):
        if self.num_excluded_abstracts > 0:
            self.num_excluded_abstracts -= 1
            return "excluded"
        elif self.num_included_abstracts > 0:
            self.num_included_abstracts -= 1
            return "included"
        else:
            raise ValueError("No more abstracts to consume")

    def default_rng(self, seed):
        self.seed_generator = np.random.default_rng(seed)

    def default_rng_few_shot(self):
        """
        Used to ensure few_shot generation is consistent across runs
        """
        return np.random.default_rng(self.seed_generator.integers(0, 2**32 - 1))

    def check_completion(self):
        # Currently, failures not thrown during text generation completely stop execution.
        if self.result_queue.len() != self.total:
            raise ValueError(
                "For some unknown reason there was an error processing the correct number of chain of thoughts"
            )

    def cleanup(self):
        self.excluded_abstracts = None
        self.included_abstracts = None
        self.gpt_cot = None

class KillException(Exception):
    pass

class Item(BaseModel):
    include_samples: int
    exclude_samples: int
    model: str = "gemini-pro"
    preprompt: str
    prompt: str

    gpt_cot_id: str = None
    few_shot_exclude: int = 0
    few_shot_include: int = 0

    include_dataset: str
    exclude_dataset: str

    seed: int = 1


class COTItem(BaseModel):
    include_samples: int
    exclude_samples: int
    model: str = "gemini-pro"
    preprompt: str
    prompt: str

    include_dataset: str
    exclude_dataset: str

    seed: int = 1


class Result(BaseModel):
    prompt: str
    llm_answer: Union[str, None]

    correct: bool
    skipped: bool

    error: Union[str, None] = None
    predicted_value: Union[str, None] = ""
    actual_value: str = ""


class COTResult(BaseModel):
    prompt: str
    llm_answer: Union[str, None]

    correct: bool
    skipped: bool

    error: Union[str, None] = None
    predicted_value: Union[str, None] = ""
    actual_value: str = ""
    test_abstract: str = ""


@stub.function()
@modal.web_endpoint(label="submit", method="POST")
def f(item: Item):
    correct_count = 0
    completed_count = 0
    skipped_count = 0
    results = []
    mode = "test"
    
    stub.status_tracker[mode + "kill"] = False

    total_prompts = item.include_samples + item.exclude_samples

    generation_task = GenerationTask(
        mode,
        item.exclude_samples,
        item.include_samples,
        item.preprompt,
        item.prompt,
        item.model,
        item.include_dataset,
        item.exclude_dataset,
        gpt_cot_id=item.gpt_cot_id,
        few_shot_exclude=item.few_shot_exclude,
        few_shot_include=item.few_shot_include,
    )

    try:
        gen.remote(
            item.seed,
            generation_task,
        )
    except KillException:
        stub.status_tracker[mode + "kill"] = False
        from fastapi import HTTPException
        raise HTTPException(status_code=499, detail="User killed operation")

    generation_task.check_completion()

    for res in stub.result_queue.get_many(total_prompts):
        if res["skipped"]:
            skipped_count += 1
        else:
            if res["correct"]:
                correct_count += 1
            completed_count += 1

        results.append(
            Result(
                prompt=res["prompt"],
                llm_answer=res.get("llm_answer"),
                correct=res["correct"],
                skipped=res["skipped"],
                predicted_value=res["predicted_value"],
                actual_value=res["actual_value"],
                error=str(res["error"]) if res["skipped"] else None,
            )
        )

    status_tracker: StatusTracker = stub.status_tracker[mode]
    stub.status_tracker[mode] = StatusTracker(mode)

    return {
        "results": results,
        "total_correct": correct_count,
        "total": completed_count,
        "total_skipped": skipped_count,
        "status_tracker": status_tracker.__dict__,
    }


@stub.function(volumes={MOUNT_DIR_COT: cot_vol})
@modal.web_endpoint(label="gptcot", method="POST")
async def gptcot(item: COTItem):
    correct_count = 0
    completed_count = 0
    skipped_count = 0
    results = []
    mode = "gptcot"
    
    stub.status_tracker[mode + "kill"] = False

    num_excluded_abstracts = item.exclude_samples
    num_included_abstracts = item.include_samples

    total_prompts = num_excluded_abstracts + num_included_abstracts

    generation_task = GenerationTask(
        mode,
        num_excluded_abstracts,
        num_included_abstracts,
        item.preprompt,
        item.prompt,
        item.model,
        item.include_dataset,
        item.exclude_dataset,
    )

    try:
        gen.remote(
            item.seed,
            generation_task,
        )
    except KillException:
        stub.status_tracker[mode + "kill"] = False
        from fastapi import HTTPException
        raise HTTPException(status_code=499, detail="User killed operation")

    generation_task.check_completion()

    for res in generation_task.result_queue.get_many(total_prompts):
        if res["skipped"]:
            skipped_count += 1
        else:
            if res["correct"]:
                correct_count += 1
            completed_count += 1

        results.append(
            COTResult(
                prompt=res["prompt"],
                llm_answer=res.get("llm_answer"),
                correct=res["correct"],
                skipped=res["skipped"],
                predicted_value=res["predicted_value"],
                actual_value=res["actual_value"],
                test_abstract=res["test_abstract"],
                error=str(res["error"]) if res["skipped"] else None,
            )
        )

    status_tracker = stub.status_tracker[mode]
    stub.status_tracker[mode] = StatusTracker(mode)
    
    unique_id = generate_unique_id()

    def filter_and_save_results(unique_id, results):
        # filter each of the results.
        # Save ones that are correct, not skipped, isolate for llm_answer with abstract
        # {abstract: test_abstract, llm_answer: llm_answer, actual_value: actual_value}
        # all under file name unique_id.csv
        import pandas as pd

        new_results = [
            [
                result.actual_value,
                result.test_abstract,
                result.llm_answer,
                result.prompt,
            ]
            for result in results
            if result.correct and not result.skipped
        ]
        df = pd.DataFrame(
            new_results,
            columns=["actual_value", "abstract", "chain_of_thought", "final_prompt"],
        )
        csv_file_path = f"{MOUNT_DIR_COT}/cot_{unique_id}.csv"
        df.to_csv(csv_file_path, index=True)
        cot_vol.commit()

    filter_and_save_results(unique_id, results)

    print("Completed and saved as ", unique_id)

    return {
        "results": results,
        "total_correct": correct_count,
        "total": completed_count,
        "total_skipped": skipped_count,
        "status_tracker": status_tracker.__dict__,
        "total_exclude_prompts": num_excluded_abstracts,
        "total_include_prompts": num_included_abstracts,
        "model": item.model,
        "id": unique_id,
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
                        seed=TEXT_SEED,
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


@stub.function(
    volumes={MOUNT_DIR: vol, MOUNT_DIR_COT: cot_vol},
    secrets=[
        modal.Secret.from_name("srma-openai"),
        modal.Secret.from_name("srma-gemini"),
    ],
    memory=1024,
    cpu=2.0,
)
async def gen(
    seed: int,
    generation_task: GenerationTask,
):
    import asyncio
    import logging

    cot_vol.reload()

    ### SLOW Reading from disk
    generation_task.load_abstracts()
    generation_task.load_gpt_cot()
    generation_task.default_rng(seed)
    ### SLOW Reading from disk

    # initialize logging
    logging_level = logging.INFO
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # it seems like gemini-pro has a rate limit that updates every half minute
    seconds_to_pause_after_rate_limit_error = (
        30 if generation_task.model == "gemini-pro" else 15
    )
    seconds_to_sleep_each_loop = (
        0.001  # 1 ms limits max throughput to 1,000 requests per second
    )
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()
    status_tracker = StatusTracker(generation_task.mode)
    next_request = None  # variable to hold the next request to call
    # initialize available capacity counts
    max_requests_per_minute = USAGE_LIMITS[generation_task.model][
        "max_requests_per_minute"
    ]
    max_tokens_per_minute = USAGE_LIMITS[generation_task.model]["max_tokens_per_minute"]
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    while True:
        if next_request is None:
            if generation_task.count():
                logging.info(f"Number of prompts left left: {generation_task.count()}")
                next_request = Abstract(
                    task_id=next(task_id_generator),
                    generation_task=generation_task,
                    attempts_left=3,
                )
                next_request.setup()
                status_tracker.start()
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

        will_die = stub.status_tracker[generation_task.mode + "kill"]
        if will_die:
            raise KillException("Killed")
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
            # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
            logging.warn(
                f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
            )
            await asyncio.sleep(remaining_seconds_to_pause)
    if status_tracker.num_tasks_failed > 0:
        logging.warning(
            f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors should be returned."
        )
    if status_tracker.num_rate_limit_errors > 0:
        logging.warning(
            f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
        )

    stub.status_tracker[generation_task.mode] = status_tracker

    generation_task.cleanup()
    return 1


@stub.function(
    volumes={MOUNT_DIR: vol},
)
@modal.web_endpoint(label="ls", method="GET")
def ls():
    import glob

    return [
        {"name": os.path.basename(file), "size": os.path.getsize(file)}
        for file in glob.glob(f"{MOUNT_DIR}/*.csv")
    ]


### PARALLEL PROCESSING HELPERS ##########################################


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    mode: str
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits

    def start(self):
        self.num_tasks_started += 1
        self.num_tasks_in_progress += 1
        stub.status_tracker[self.mode] = self
    def error(self):
        self.num_other_errors += 1
        stub.status_tracker[self.mode] = self
    def fail(self):
        self.num_tasks_in_progress -= 1
        self.num_tasks_failed += 1
        stub.status_tracker[self.mode] = self
    def success(self):
        self.num_tasks_in_progress -= 1
        self.num_tasks_succeeded += 1
        stub.status_tracker[self.mode] = self
        

@dataclass
class Abstract:
    task_id: int
    attempts_left: int
    generation_task: GenerationTask

    actual_value: str = None
    final_prompt: str = None
    result: list = field(default_factory=list)
    token_consumption: int = 0
    abstract: str = ""

    def sample_abstract(self, actual_value, seed_generator, n=1):

        abstracts = (
            self.generation_task.get_abstract_series(actual_value=actual_value)
            .sample(n=n, random_state=seed_generator)
            .tolist()
        )

        return abstracts

    def sample_cot_abstract(self, actual_value, seed_generator, n=1):
        searchable_abstracts = self.generation_task.get_cot_abstract_dataframe(
            actual_value=actual_value
        )
        if len(searchable_abstracts) < n:
            raise ValueError(
                f"Insufficient number of abstracts to sample from. {len(searchable_abstracts)} < {n}"
            )
        abstracts = searchable_abstracts.sample(
            n=n, random_state=seed_generator
        ).to_dict("records")

        return abstracts

    def setup(self):
        if self.final_prompt:
            return
        self.actual_value = self.generation_task.consume()
        # Get a random abstract
        self.abstract = self.sample_abstract(
            self.actual_value, self.generation_task.seed_generator, n=1
        )[0]

        seed_generator_few_shot = self.generation_task.default_rng_few_shot()

        few_shot_text = ""
        if self.generation_task.needs_append_few_shots():
            if not self.generation_task.is_generated_few_shot():
                few_shot_text += "\nExamples:\n"
                number_include = self.generation_task.few_shot_include
                number_exclude = self.generation_task.few_shot_exclude

                few_shot_exclusionary_abstracts = self.sample_abstract(
                    "excluded", seed_generator_few_shot, n=number_exclude
                )
                few_shot_inclusionary_abstracts = self.sample_abstract(
                    "included", seed_generator_few_shot, n=number_include
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
            elif self.generation_task.is_generated_few_shot():
                few_shot_text += "\n# Examples\n"
                number_include = self.generation_task.few_shot_include
                number_exclude = self.generation_task.few_shot_exclude

                few_shot_exclusionary_abstracts = self.sample_cot_abstract(
                    "excluded", seed_generator_few_shot, n=number_exclude
                )
                few_shot_inclusionary_abstracts = self.sample_cot_abstract(
                    "included", seed_generator_few_shot, n=number_include
                )

                for abstract in few_shot_exclusionary_abstracts:
                    few_shot_text += f"""
    ## Example Abstract
    {abstract["abstract"]}

    ## Rationale and Conclusion
    {abstract["chain_of_thought"]}
    """
                for abstract in few_shot_inclusionary_abstracts:
                    few_shot_text += f"""
    ## Example Abstract
    {abstract["abstract"]}

    ## Rationale and Conclusion
    {abstract["chain_of_thought"]}
    """
                few_shot_text += "\n# End of Examples\n"
            else:
                raise ValueError("Invalid few shot generation")

        # Create the prompt and get the response
        self.final_prompt = FINAL_PROMPT.format(
            preprompt=self.generation_task.preprompt,
            few_shot_text=few_shot_text,
            test_abstract=self.abstract,
            prompt=self.generation_task.prompt,
        )
        self.token_consumption = calculate_tokens(
            self.generation_task.model, self.final_prompt, MAX_COMPLETION_TOKENS
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
            model_async = setup.local(self.generation_task.model)

            # Model produces response
            llm_answer = await model_async(self.final_prompt)

            # Evaluate response
            predicted_value = is_excluded(llm_answer, self.final_prompt)
            ###### BUSINESS
        except Exception as e:
            if "Rate limit" in str(e) or "429" in str(e):
                status_tracker.time_of_last_rate_limit_error = time.time()

            status_tracker.error()
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                self.generation_task.result_queue.put(
                    {
                        "skipped": True,
                        "prompt": self.final_prompt,
                        "error": [str(e) for e in self.result],
                        "correct": False,
                        "predicted_value": "fail",
                        "actual_value": "fail",
                        "llm_answer": "fail",
                        "test_abstract": self.abstract,
                    }
                )
                status_tracker.fail()
        else:
            self.generation_task.result_queue.put(
                {
                    "skipped": False,
                    "prompt": self.final_prompt,
                    "error": None,
                    "correct": predicted_value == self.actual_value,
                    "predicted_value": predicted_value,
                    "actual_value": self.actual_value,
                    "llm_answer": llm_answer,
                    "test_abstract": self.abstract,
                }
            )
            status_tracker.success()


##### GENERAL HELPERS ######################################################


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


def generate_unique_id():
    import uuid

    unique_id = str(uuid.uuid4())[:8]
    return unique_id


@stub.function(
    volumes={MOUNT_DIR: vol},
)
def create_subsets():
    import os
    import pandas as pd

    files = os.listdir(MOUNT_DIR)
    files = [file for file in files if file.endswith(".csv")]

    # split into train and test sets, use 1k from each file for train, 1k for validation, and the rest for testing
    trues = list(create_subset.map(files))
    print(trues, os.listdir(MOUNT_DIR))
    return True


@stub.function(
    volumes={MOUNT_DIR: vol},
)
def create_subset(file):
    import os
    import pandas as pd

    print("Processing ", file)
    df = pd.read_csv(os.path.join(MOUNT_DIR, file), low_memory=False)
    # remove .csv from end of filename using os
    file = os.path.splitext(file)[0]
    train = df.sample(1000)
    df = df.drop(train.index)
    validation = df.sample(1000)
    test = df.drop(validation.index)
    train.to_csv(os.path.join(MOUNT_DIR, f"{file}_train_1000.csv"), index=False)
    print("Train saved as ", f"{file}_train_1000.csv")
    validation.to_csv(
        os.path.join(MOUNT_DIR, f"{file}_validation_1000.csv"), index=False
    )
    print("Validation saved as ", f"{file}_validation_1000.csv")
    test.to_csv(os.path.join(MOUNT_DIR, f"{file}_test.csv"), index=False)
    print("Test saved as ", f"{file}_test.csv")
    vol.commit()
    return True

# @stub.function(
#     volumes={MOUNT_DIR: vol},
# )
# def remove_files():
#     import os
#     import glob
#     # glob pattern = *.csv_*
#     glob_pattern = os.path.join(MOUNT_DIR, "*.csv_*")
#     files = glob.glob(glob_pattern)
#     print(files)
#     for file in files:
#         os.remove(os.path.join(MOUNT_DIR, file))
#     vol.commit()
#     return True

@stub.local_entrypoint()
def main():
    create_subsets.remote()
