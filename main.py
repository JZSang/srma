# Modal
import modal

# System
import os
import time
import asyncio
from typing import Literal, Optional, Union
from dataclasses import dataclass, field

# Typing
from pydantic import BaseModel
from fastapi import UploadFile, File, Body

# Source
from modal_references import vol_dataset, vol_save_results, cot_vol, stub
from datasets.upload import dataset_upload_impl
from datasets.manage import dataset_manage_wrapper, DatasetManage
from operations.kill import kill_impl, KillException
from operations.status import get_status_impl
from tasks.generation_task import GenerationTask
from constants import (
    MOUNT_DIR,
    MOUNT_DIR_COT,
    TOKENIZER,
    MAX_COMPLETION_TOKENS,
    USAGE_LIMITS,
)

# Prompt format
FINAL_PROMPT = "{preprompt}\n{few_shot_text}\n# Abstract in investigation: \n{test_abstract}\n\n{prompt}\n"


UploadOperations = Literal["dataset_upload"]

@stub.function()
@modal.web_endpoint(label="router-upload", method="POST")
async def router_upload(operation: UploadOperations, file: UploadFile = File(...)):
    if operation == "dataset_upload":
        return await dataset_upload(file)
    else:
        raise ValueError("Invalid operation")

Operations = Literal["dataset_manage", "status", "kill"]
class Routing(BaseModel):    
    dataset_manage: DatasetManage = None
    status: str = None
    kill: str = None

@stub.function()
@modal.web_endpoint(label="router", method="POST")
async def router(routing: Routing, operation: Operations):
    print("CANONICAL-API-LINE: ", operation, routing.dict())
    if operation == "dataset_manage":
        return dataset_manage(routing.dataset_manage)
    elif operation == "status":
        return get_status(routing.status)
    elif operation == "kill":
        return kill(routing.kill)
    else:
        raise ValueError("Invalid operation")

async def dataset_upload(file: UploadFile = File(...)):
    if file.content_type != "text/csv":
        raise ValueError("File must be a CSV")
    content = await file.read()
    return dataset_upload_impl.remote(content, filename=file.filename)


def dataset_manage(params: DatasetManage):
    ret = dataset_manage_wrapper.remote(params)
    return ret

def get_status(mode):
    return get_status_impl(mode)


def kill(mode):
    return kill_impl(mode)


class Item(BaseModel):
    abstract_in_investigation: str = None
    abstract_in_investigation_actual_value: str = None

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
    ensemble: int = 1


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
    predicted_values: list[str] = []
    actual_value: str = ""
    test_abstract: str = ""

    token_counts: dict = {}


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
    call = run_f.spawn(item)
    print("Starting run with ", call.object_id)
    return {"function_id": call.object_id}


@stub.function()
@modal.web_endpoint(label="check-test", method="GET")
def check_test(function_id=None):
    if not function_id:
        raise ValueError("Invalid function_id")
    run_f_function = modal.functions.FunctionCall.from_id(function_id)
    try:
        return run_f_function.get(timeout=0)
    except TimeoutError:
        return {"function_id": run_f_function.object_id}
    except:
        raise ValueError("Invalid state")


@stub.function(timeout=60 * 60)
def run_f(item: Item):
    import time

    start_time = time.time()
    correct_count = 0
    completed_count = 0
    skipped_count = 0
    results = []
    mode = "test"

    stub.status_tracker[mode + "kill"] = False

    generation_tasks: list[GenerationTask] = []
    for model_seed in range(item.ensemble):
        generation_tasks.append(
            GenerationTask(
                mode,
                item.exclude_samples,
                item.include_samples,
                item.preprompt,
                item.prompt,
                item.model,
                model_seed,
                item.include_dataset,
                item.exclude_dataset,
                gpt_cot_id=item.gpt_cot_id,
                few_shot_exclude=item.few_shot_exclude,
                few_shot_include=item.few_shot_include,
                abstract_in_investigation=item.abstract_in_investigation,
                abstract_in_investigation_actual_value=item.abstract_in_investigation_actual_value,
            )
        )

    try:
        generation_tasks = gen.remote(
            item.seed,
            generation_tasks,
        )
    except KillException:
        stub.status_tracker[mode + "kill"] = False

    import tiktoken

    for i, generation_task in enumerate(generation_tasks):
        generation_task.check_completion(i)

    from collections import defaultdict

    total_result_queue = [
        result_raw
        for generation_task in generation_tasks
        for result_raw in generation_task.get_all_result_queue()
    ]
    result_queue_dict = defaultdict(list)
    for result_raw in total_result_queue:
        result_queue_dict[result_raw["abstract_id"]].append(result_raw)

    for res_key in result_queue_dict:
        from collections import Counter

        res_tuple = result_queue_dict[res_key]

        # Aggregate stats
        is_skipped = 0
        index_of_skipped = -1

        for i, res in enumerate(res_tuple):
            if res["skipped"]:
                is_skipped = 1
                index_of_skipped = i
                break
        skipped_count += is_skipped
        completed_count += not is_skipped

        # Token counting
        encoding = tiktoken.get_encoding(TOKENIZER)
        appended_llm_answers = [res.get("llm_answer") or "" for res in res_tuple]
        llm_answers_token_counts = sum(
            [
                len(encoding.encode(appended_llm_answer))
                for appended_llm_answer in appended_llm_answers
            ]
        )
        prompt_token_count = len(encoding.encode(res["prompt"]))
        token_counts = {
            "llm_answer": llm_answers_token_counts,
            "prompt": prompt_token_count,
            "prompt_total": prompt_token_count * item.ensemble,
        }

        prompt = res_tuple[0]["prompt"]
        llm_answer = "\n---\n".join(appended_llm_answers)
        skipped = True if is_skipped else False
        predicted_value = Counter(
            [res["predicted_value"] for res in res_tuple]
        ).most_common(1)[0][0]
        actual_value = res_tuple[0]["actual_value"]
        correct = predicted_value == actual_value
        error = str(res_tuple[index_of_skipped]["error"]) if is_skipped else None
        test_abstract = res_tuple[0]["test_abstract"]
        token_counts = token_counts
        predicted_values = [res["predicted_value"] for res in res_tuple]

        correct_count += correct if not is_skipped else 0

        results.append(
            Result(
                prompt=prompt,
                llm_answer=llm_answer,
                correct=correct,
                skipped=skipped,
                predicted_value=predicted_value,
                actual_value=actual_value,
                error=error,
                test_abstract=test_abstract,
                token_counts=token_counts,
                predicted_values=predicted_values,
            )
        )

    status_tracker: StatusTracker = stub.status_tracker[mode]
    stub.status_tracker[mode] = StatusTracker(mode)

    print("Completed, status: ", status_tracker.__dict__)
    end_time = time.time()
    print("Time taken: ", end_time - start_time)

    final_results = {
        "model": item.model,
        "results": results,
        "total_correct": correct_count,
        "total": completed_count,
        "total_skipped": skipped_count,
        "status_tracker": status_tracker.__dict__,
        "include_dataset": item.include_dataset,
        "exclude_dataset": item.exclude_dataset,
        "few_shot_exclude": item.few_shot_exclude,
        "few_shot_include": item.few_shot_include,
        "ensemble": item.ensemble,
    }

    save_final_results.spawn(final_results, time=int(time.time()))

    return final_results


@stub.function(volumes={MOUNT_DIR: vol_save_results})
async def save_final_results(final_results, time):
    final_results["results"] = [result.dict() for result in final_results["results"]]
    with open(os.path.join(MOUNT_DIR, f"results_{time}.json"), "w") as f:
        import json

        json.dump(final_results, f)
    vol_save_results.commit()


@stub.function(volumes={MOUNT_DIR_COT: cot_vol}, timeout=600)
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
        1,
        item.include_dataset,
        item.exclude_dataset,
    )

    try:
        generation_task = gen.remote(
            item.seed,
            generation_task,
        )
    except KillException:
        stub.status_tracker[mode + "kill"] = False

    if isinstance(generation_task, list):
        generation_task = generation_task[0]
    generation_task.check_completion()

    for res in generation_task.get_all_result_queue():
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
        number_used_included = len(df[df["actual_value"] == "included"])
        number_used_excluded = len(df[df["actual_value"] == "excluded"])
        csv_file_path = f"{MOUNT_DIR_COT}/cot_{unique_id}.csv"
        df.to_csv(csv_file_path, index=True)
        cot_vol.commit()
        return number_used_excluded, number_used_included

    number_used_excluded, number_used_included = filter_and_save_results(
        unique_id, results
    )

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
        "include_dataset": item.include_dataset,
        "exclude_dataset": item.exclude_dataset,
        "number_used_excluded": number_used_excluded,
        "number_used_included": number_used_included,
    }


@stub.function()
def setup(model, model_seed):
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
                (
                    await client.chat.completions.create(
                        messages=generate_messages(model, prompt),
                        model=model,
                        max_tokens=MAX_COMPLETION_TOKENS,
                        seed=model_seed,
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
    else:
        raise ValueError("Invalid model calculate tokens")


def is_excluded(text, test_abstract):
    if text is None:
        raise Exception(f"Invalid case: LLM did not print an answer: {test_abstract}")
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
        or model == "gpt-4-turbo-2024-04-09"
    ):
        return [{"role": "user", "content": prompt}]
    else:
        raise ValueError(f"Invalid model generate_messages {model}")


@stub.function(
    volumes={MOUNT_DIR: vol_dataset, MOUNT_DIR_COT: cot_vol},
    secrets=[
        modal.Secret.from_name("srma-openai"),
        modal.Secret.from_name("srma-gemini"),
    ],
    memory=2048,
    cpu=2.0,
    timeout=60 * 60 * 2,
)
async def gen(
    seed: int,
    generation_tasks: Union[GenerationTask, list[GenerationTask]],
):
    import asyncio
    import logging

    if not isinstance(generation_tasks, list):
        generation_tasks = [generation_tasks]

    # initialize logging
    logging_level = logging.INFO
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    seconds_to_sleep_each_loop = (
        0.007  # 7 ms limits max throughput to 142 requests per second
    )
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()

    next_request = None  # variable to hold the next request to call

    for generation_task in generation_tasks:
        ### SLOW Reading from disk
        cot_vol.reload()
        generation_task.default_rng(seed)
        generation_task.load_abstracts()
        generation_task.load_gpt_cot()
        ### SLOW Reading from disk

    i = 0
    generation_task = generation_tasks[i]
    status_tracker = StatusTracker(generation_task.mode)
    # it seems like gemini-pro has a rate limit that updates every half minute
    seconds_to_pause_after_rate_limit_error = (
        30 if generation_task.model == "gemini-pro" else 15
    )

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
            if not generation_task.count() and i < len(generation_tasks) - 1:
                i += 1
                generation_task.cleanup()
                generation_task = generation_tasks[i]
            if generation_task.count():
                logging.info(
                    f"Number of prompts left in generation_task {i}: {generation_task.count()}"
                )
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
    return generation_tasks


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
    abstract_id: int = 1

    def sample_cot_abstract(self, actual_value, seed_generator, n=1):
        searchable_abstracts = self.generation_task.get_cot_abstract_dataframe(
            actual_value=actual_value, n=n, seed_generator=seed_generator
        )
        if len(searchable_abstracts) < n:
            raise ValueError(
                f"Insufficient number of abstracts to sample from. {len(searchable_abstracts)} < {n}"
            )
        abstracts = searchable_abstracts.to_dict("records")

        return abstracts

    def setup(self):
        if self.final_prompt:
            return
        self.actual_value = self.generation_task.consume()

        if self.actual_value == "excluded":
            self.abstract = next(self.generation_task.get_next_abstract_excluded)
        else:
            self.abstract = next(self.generation_task.get_next_abstract_included)

        seed_generator_few_shot = self.generation_task.new_default_rng()

        self.abstract_id = hash(self.abstract)

        few_shot_text = ""
        if self.generation_task.needs_append_few_shots():
            if not self.generation_task.is_generated_few_shot():
                raise ValueError(
                    "do not use this type of few shot anymore, use cot in some form"
                )
            #             few_shot_text += "\n# Start of Examples\n"
            #             number_include = self.generation_task.few_shot_include
            #             number_exclude = self.generation_task.few_shot_exclude

            #             few_shot_exclusionary_abstracts = self.sample_abstract(
            #                 "excluded", seed_generator_few_shot, n=number_exclude, mode="fewshot"
            #             )
            #             few_shot_inclusionary_abstracts = self.sample_abstract(
            #                 "included", seed_generator_few_shot, n=number_include, mode="fewshot"
            #             )

            #             for abstract in few_shot_exclusionary_abstracts:
            #                 few_shot_text += f"""
            # ---
            # {abstract}

            # This article should be excluded.
            # """
            #             for abstract in few_shot_inclusionary_abstracts:
            #                 few_shot_text += f"""
            # ---
            # {abstract}

            # This article should be included.

            # """
            #             few_shot_text += "\n# End of Examples\n"
            elif self.generation_task.is_generated_few_shot():
                few_shot_text += "\n# Start of Examples\n"
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
            model_async = setup.local(
                self.generation_task.model, self.generation_task.model_seed
            )
            import asyncio

            # Model produces response
            llm_answer = await asyncio.wait_for(
                model_async(self.final_prompt), timeout=120
            )

            # Evaluate response
            predicted_value = is_excluded(llm_answer, self.final_prompt)
            ###### BUSINESS
        except (Exception, TimeoutError) as e:
            if "Rate limit" in str(e) or "429" in str(e):
                status_tracker.time_of_last_rate_limit_error = time.time()

            status_tracker.error()
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                self.generation_task.put_result_queue(
                    {
                        "skipped": True,
                        "prompt": self.final_prompt,
                        "error": [str(e) for e in self.result],
                        "correct": False,
                        "predicted_value": "fail",
                        "actual_value": "fail",
                        "llm_answer": "fail",
                        "test_abstract": self.abstract,
                        "abstract_id": self.abstract_id,
                    }
                )
                status_tracker.fail()
        else:
            self.generation_task.put_result_queue(
                {
                    "skipped": False,
                    "prompt": self.final_prompt,
                    "error": None,
                    "correct": predicted_value == self.actual_value,
                    "predicted_value": predicted_value,
                    "actual_value": self.actual_value,
                    "llm_answer": llm_answer,
                    "test_abstract": self.abstract,
                    "abstract_id": self.abstract_id,
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


@stub.local_entrypoint()
def main():
    print(stub.file_lock.get("lock"))
