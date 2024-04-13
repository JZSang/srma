import modal
import os
import time
from pydantic import BaseModel
from dataclasses import dataclass, field
from fastapi import UploadFile, File
from typing import Union
import asyncio
import numpy as np

# Constant texts
FINAL_PROMPT = "{preprompt}\n{few_shot_text}\n# Abstract in investigation: \n{test_abstract}\n\n{prompt}\n"
FILTER_EMPTY_COLUMNS = ["Abstract", "Content"]

### LLM SETTINGS
MAX_COMPLETION_TOKENS = 2048
USAGE_LIMITS = {
    "gpt-3.5-turbo": {
        "max_requests_per_minute": 10000,
        "max_tokens_per_minute": 2000000,
    },
    "gpt-4-0125-preview": {
        "max_requests_per_minute": 10000,
        "max_tokens_per_minute": 1500000,
    },
    "gpt-4-turbo-2024-04-09": {
        "max_requests_per_minute": 10000,
        "max_tokens_per_minute": 1500000,
    },
    "gpt-3.5-turbo-0125": {
        "max_requests_per_minute": 10000,
        "max_tokens_per_minute": 2000000,
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
# vol_dataset = modal.Volume.persisted("srma")
vol_dataset = modal.Volume.persisted("srma-dataset")
MOUNT_DIR = "/data"
EXCLUDED_ABSTRACTS = "excl_final.csv"
INCLUDED_ABSTRACTS = "incl_final.csv"


### GPTCoT storage
MOUNT_DIR_COT = "/cot"
cot_vol = modal.Volume.persisted("srma-cot")

vol_save_results = modal.Volume.persisted("srma-results")


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

stub.file_lock = modal.Dict.new()
stub.file_metadata_queue = modal.Queue.new()


class Metadata(BaseModel):
    lines: int
    size: int
    columns: list[str]
    filtered: bool
    file: str
    deleted: bool
    original_file: str
    split_type: str


@stub.function(volumes={MOUNT_DIR: vol_dataset})
@modal.web_endpoint(label="dataset-upload", method="POST")
async def dataset_upload(file: UploadFile = File(...)):
    if file.content_type != "text/csv":
        raise ValueError("File must be a CSV")

    import pandas as pd
    import numpy as np
    from io import StringIO

    # 1. Save the file
    content = await file.read()
    data = content.decode("utf-8")
    df = pd.read_csv(StringIO(data), low_memory=False)
    df = df.fillna(np.nan)
    for col in FILTER_EMPTY_COLUMNS:
        if col in df.columns:
            df = df.dropna(subset=[col])
    df.to_csv(os.path.join(MOUNT_DIR, file.filename), index=False)

    # Add metadata to be saved into a queue. Will call on page load.
    info = Metadata(
        split_type="Raw",
        lines=len(df),
        size=os.path.getsize(os.path.join(MOUNT_DIR, file.filename)),
        columns=list(df.columns),
        filtered=False,
        file=file.filename,
        deleted=False,
        original_file=file.filename,
    )
    stub.file_metadata_queue.put(info)

    # 2. Commit
    vol_dataset.commit()

    # 3. Show preview
    return info


class DatasetManage(BaseModel):
    type: str = "ls"
    file: str = None
    files: list[str] = None

    column: str = None

    splits: tuple[int, int, int, int] = None


@stub.function(volumes={MOUNT_DIR: vol_dataset})
@modal.web_endpoint(label="dataset-manage", method="POST")
async def dataset_manage(params: DatasetManage):
    # Pause operation until lock is released
    while stub.file_lock.get("lock", False):
        await asyncio.sleep(0.1)
    stub.file_lock["lock"] = True
    try:
        ret = await dataset_manage_impl(params)
    finally:
        stub.file_lock["lock"] = False
    return ret


async def dataset_manage_impl(params: DatasetManage):
    if params.type == "delete":
        if not params.files:
            return True
        import os

        for file in params.files:
            os.remove(os.path.join(MOUNT_DIR, file))
            # Make sure there's if statement to check deleted
            # Make sure that deleted instances throw an error
            stub.file_metadata_queue.put(
                Metadata(
                    split_type="Raw",
                    lines=0,
                    size=0,
                    columns=[],
                    filtered=False,
                    file=file,
                    deleted=True,
                    original_file=file,
                )
            )
        vol_dataset.commit()

        return True
    elif params.type == "ls":
        import os

        files: list[Metadata] = stub.file_metadata_queue.get_many(9999, block=False)
        if files:
            vol_dataset.reload()
            with open(os.path.join(MOUNT_DIR, "metadata.json"), "r") as f:
                import json

                data = json.load(f)
            with open(os.path.join(MOUNT_DIR, "metadata.json"), "w") as f:
                for file in files:
                    if file.deleted:
                        del data[file.file]
                        continue
                    data[file.file] = file.dict()
                json.dump(data, f)

            vol_dataset.commit()
        with open(os.path.join(MOUNT_DIR, "metadata.json"), "r") as f:
            import json

            content = json.load(f)
            return [content[file] for file in sorted(content.keys())]

    elif params.type == "filter_na":
        import pandas as pd

        vol_dataset.reload()
        df = pd.read_csv(os.path.join(MOUNT_DIR, params.file))
        df = df.dropna(subset=[params.column])
        df.to_csv(os.path.join(MOUNT_DIR, params.file), index=False)

        stub.file_metadata_queue.put(
            Metadata(
                lines=len(df),
                size=os.path.getsize(os.path.join(MOUNT_DIR, params.file)),
                columns=list(df.columns),
                filtered=True,
                file=params.file,
                deleted=False,
                original_file=params.file,
            )
        )

        vol_dataset.commit()
        return {"remaining_empty": df[params.column].isna().sum()}

    elif params.type == "split":
        import pandas as pd
        import numpy as np
        import os

        np.random.seed(1337)

        # validate
        gptcot, train, val, test = params.splits

        vol_dataset.reload()
        df = pd.read_csv(os.path.join(MOUNT_DIR, params.file), low_memory=False)
        if gptcot + train + val + test != len(df):
            raise ValueError("Invalid split sizes")

        # create dataframe splits
        df_gptcot = df.sample(gptcot)
        df = df.drop(df_gptcot.index)

        df_train = df.sample(train)
        df = df.drop(df_train.index)

        df_val = df.sample(val)
        df = df.drop(df_val.index)

        df_test = df.sample(test)
        df = df.drop(df_test.index)

        filename = os.path.splitext(params.file)[0]

        # save to disk
        gptcot_filename = f"{filename}_gen_{gptcot}_gptcot.csv"
        train_filename = f"{filename}_gen_{train}_train.csv"
        val_filename = f"{filename}_gen_{val}_val.csv"
        test_filename = f"{filename}_gen_{test}_test.csv"
        df_gptcot.to_csv(os.path.join(MOUNT_DIR, gptcot_filename), index=False)
        df_train.to_csv(os.path.join(MOUNT_DIR, train_filename), index=False)
        df_val.to_csv(os.path.join(MOUNT_DIR, val_filename), index=False)
        df_test.to_csv(os.path.join(MOUNT_DIR, test_filename), index=False)

        # generate metadata
        info_gptcot = Metadata(
            split_type="GPTCoT",
            lines=len(df_gptcot),
            size=os.path.getsize(os.path.join(MOUNT_DIR, gptcot_filename)),
            columns=list(df_gptcot.columns),
            filtered=True,
            file=gptcot_filename,
            deleted=False,
            original_file=params.file,
        )
        info_train = Metadata(
            split_type="Train",
            lines=len(df_train),
            size=os.path.getsize(os.path.join(MOUNT_DIR, train_filename)),
            columns=list(df_train.columns),
            filtered=True,
            file=train_filename,
            deleted=False,
            original_file=params.file,
        )
        info_val = Metadata(
            split_type="Validation",
            lines=len(df_val),
            size=os.path.getsize(os.path.join(MOUNT_DIR, val_filename)),
            columns=list(df_val.columns),
            filtered=True,
            file=val_filename,
            deleted=False,
            original_file=params.file,
        )
        info_test = Metadata(
            split_type="Test",
            lines=len(df_test),
            size=os.path.getsize(os.path.join(MOUNT_DIR, test_filename)),
            columns=list(df_test.columns),
            filtered=True,
            file=test_filename,
            deleted=False,
            original_file=params.file,
        )
        stub.file_metadata_queue.put_many(
            [info_gptcot, info_train, info_val, info_test]
        )

        # commit to disk
        vol_dataset.commit()

        return {
            "gptcot": len(df_gptcot),
            "train": len(df_train),
            "val": len(df_val),
            "test": len(df_test),
        }


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
    abstract_in_investigation = None
    abstract_in_investigation_is_consumed = False

    def __init__(
        self,
        mode,
        num_excluded_abstracts,
        num_included_abstracts,
        preprompt,
        prompt,
        model,
        model_seed,
        include_dataset,
        exclude_dataset,
        gpt_cot_id=None,
        few_shot_exclude=0,
        few_shot_include=0,
        abstract_in_investigation=None,
        abstract_in_investigation_actual_value=None,
    ):
        self.mode = mode
        if mode == "test":
            self.result_queue = []
            print(
                f"Running test mode with {num_excluded_abstracts} excluded abstracts and {num_included_abstracts} included abstracts"
            )
        elif mode == "gptcot":
            self.result_queue = []
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

        assert self.length_result_queue() == 0

        if abstract_in_investigation and abstract_in_investigation_actual_value:
            self.abstract_in_investigation = abstract_in_investigation
            self.abstract_in_investigation_actual_value = (
                abstract_in_investigation_actual_value
            )

        self.num_excluded_abstracts = num_excluded_abstracts
        self.num_included_abstracts = num_included_abstracts

        self.preprompt = preprompt
        self.prompt = prompt
        self.model = model
        self.model_seed = model_seed

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
        if abstract_in_investigation or abstract_in_investigation_actual_value:
            if self.num_excluded_abstracts + self.num_included_abstracts > 0:
                raise ValueError(
                    "abstract_in_investigation should not be provided when there are abstracts to process"
                )
        elif self.total == 0:
            raise ValueError("No abstracts to process")

    def count(self):
        if self.abstract_in_investigation:
            return 1 if not self.abstract_in_investigation_is_consumed else 0
        return self.num_excluded_abstracts + self.num_included_abstracts

    def is_generated_few_shot(self):
        return bool(self.gpt_cot_id)

    def needs_append_few_shots(self):
        return self.few_shot_exclude or self.few_shot_include

    def put_result_queue(self, result):
        self.result_queue.append(result)

    def get_all_result_queue(self):
        return sorted(self.result_queue, key=lambda x: x["test_abstract"])

    def length_result_queue(self):
        return len(self.result_queue)

    def load_abstracts(self):
        import pandas as pd
        import os
        import numpy as np

        if "seed_generator" not in self.__dict__:
            raise ValueError("seed_generator not set")

        # Load up volume files in memory
        df_excluded = pd.read_csv(
            os.path.join("/data", self.exclude_dataset), keep_default_na=False
        )
        if "Abstract" in df_excluded.columns:
            excluded_abstracts = df_excluded["Abstract"].replace("", np.nan).dropna()
        else:
            excluded_abstracts = df_excluded["Content"].replace("", np.nan).dropna()

        df_included = pd.read_csv(
            os.path.join("/data", self.include_dataset), keep_default_na=False
        )
        if "Abstract" in df_included.columns:
            included_abstracts = df_included["Abstract"].replace("", np.nan).dropna()
        else:
            included_abstracts = df_included["Content"].replace("", np.nan).dropna()
        self.excluded_abstracts = excluded_abstracts.sample(
            frac=1, random_state=self.seed_generator
        )
        self.included_abstracts = included_abstracts.sample(
            frac=1, random_state=self.seed_generator
        )
        self.get_next_abstract_excluded = self.get_next_abstract_generator(True)
        self.get_next_abstract_included = self.get_next_abstract_generator(False)

    # generator
    def get_next_abstract_generator(self, exclude):
        if exclude:
            for abstract in self.excluded_abstracts:
                yield abstract
        elif not exclude:
            for abstract in self.included_abstracts:
                yield abstract
        else:
            raise ValueError(
                f"Invalid actual_value excluded={exclude}, this should never happen"
            )

    def get_abstracts(
        self, actual_value="excluded", n=1, random_state=None, mode="testset"
    ):
        if self.abstract_in_investigation:
            import pandas as pd

            return pd.Series([self.abstract_in_investigation])
        exclude = True if actual_value == "excluded" else False
        if exclude:
            samples = self.excluded_abstracts.sample(
                n=n, random_state=random_state, replace=False
            )
            # Sample without replacement
            self.excluded_abstracts = self.excluded_abstracts.drop(samples.index)
            return samples
        else:
            samples = self.included_abstracts.sample(
                n=n, random_state=random_state, replace=False
            )
            # Sample without replacement
            self.included_abstracts = self.included_abstracts.drop(samples.index)
            return samples

    def get_cot_abstract_dataframe(
        self, actual_value="excluded", n=1, seed_generator=None
    ):
        if self.gpt_cot is None:
            raise ValueError("gpt_cot is not loaded")
        return self.gpt_cot[self.gpt_cot["actual_value"] == actual_value].sample(
            n=n, random_state=seed_generator, replace=False
        )

    def load_gpt_cot(self):
        import pandas as pd
        import os

        if self.gpt_cot_id:
            self.gpt_cot = pd.read_csv(
                os.path.join(MOUNT_DIR_COT, f"cot_{self.gpt_cot_id}.csv")
            )

    def consume(self):
        if self.abstract_in_investigation:
            self.abstract_in_investigation_is_consumed = True
            return self.abstract_in_investigation_actual_value
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

    def new_default_rng(self):
        """
        Used to ensure generation is consistent across runs for subtree operations
        """
        return np.random.default_rng(self.seed_generator.integers(0, 2**32 - 1))

    def check_completion(self, i=0):
        # Currently, failures not thrown during text generation do not stop execution.
        if self.length_result_queue() != self.total:
            print(
                f"ERROR ### generation_task {i}, result_queue.len() != self.total, returning what's left anyways"
            )

    def cleanup(self):
        self.excluded_abstracts = None
        self.included_abstracts = None
        self.gpt_cot = None
        # remove all generators because they cannot be pickled
        self.get_next_abstract_excluded = None
        self.get_next_abstract_included = None


class KillException(Exception):
    pass


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
    total_result_queue = [result_raw for generation_task in generation_tasks for result_raw in generation_task.get_all_result_queue() ]
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


# @stub.function(
#     volumes={MOUNT_DIR: vol},
# )
# @modal.web_endpoint(label="ls", method="GET")
# def ls():
#     vol.reload()
#     import glob

#     return [
#         {"name": os.path.basename(file), "size": os.path.getsize(file)}
#         for file in glob.glob(f"{MOUNT_DIR}/*.csv")
#     ]


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


@stub.function(
    volumes={MOUNT_DIR: vol_dataset},
)
def create_subsets():
    import os

    # Clean all generated files
    remove_files.local()

    # Find all relevant files in directory
    files = os.listdir(MOUNT_DIR)
    files = [file for file in files if file.endswith(".csv")]

    # split into train and test sets, use 800 for gptcot, 800 for train, 800 for validation, and the rest for testing
    _ = list(create_subset.map(files))

    vol_dataset.reload()
    print(os.listdir(MOUNT_DIR))
    return True


@stub.function(
    volumes={MOUNT_DIR: vol_dataset},
)
def create_subset(file):
    import os
    import pandas as pd
    import random

    random.seed(1337)

    print("Processing ", file)
    df = pd.read_csv(os.path.join(MOUNT_DIR, file), low_memory=False)
    # remove .csv from end of filename using os
    file = os.path.splitext(file)[0]
    split = min(800, (len(df) - 400) // 3)

    if split < 800:
        split = len(df) // 2

        train = df.sample(split, random_state=1337)
        test = df.drop(train.index)

        train.to_csv(
            os.path.join(MOUNT_DIR, f"{file}_gen_train_{split}.csv"), index=False
        )
        print("Train saved as ", f"{file}_gen_train_{split}.csv")

        test.to_csv(os.path.join(MOUNT_DIR, f"{file}_gen_test.csv"), index=False)
        print("Test saved as ", f"{file}_gen_test.csv")

        vol_dataset.commit()
        return True

    gptcot_set = df.sample(split, random_state=1337)
    df = df.drop(gptcot_set.index)

    train = df.sample(split, random_state=1337)
    df = df.drop(train.index)

    validation = df.sample(split, random_state=1337)
    test = df.drop(validation.index)

    gptcot_set.to_csv(
        os.path.join(MOUNT_DIR, f"{file}_gen_gpt_cot_{split}.csv"), index=False
    )
    print("GPT CoT set saved as ", f"{file}_gen_gpt_cot_{split}.csv")

    train.to_csv(os.path.join(MOUNT_DIR, f"{file}_gen_train_{split}.csv"), index=False)
    print("Train saved as ", f"{file}_gen_train_{split}.csv")

    validation.to_csv(
        os.path.join(MOUNT_DIR, f"{file}_gen_validation_{split}.csv"), index=False
    )
    print("Validation saved as ", f"{file}_gen_validation_{split}.csv")

    test.to_csv(os.path.join(MOUNT_DIR, f"{file}_gen_test.csv"), index=False)
    print("Test saved as ", f"{file}_gen_test.csv")

    vol_dataset.commit()
    return True


@stub.function(
    volumes={MOUNT_DIR: vol_dataset},
)
def remove_files():
    import os
    import glob

    glob_pattern = os.path.join(MOUNT_DIR, "*gen*")
    files = glob.glob(glob_pattern)
    print(files)
    for file in files:
        os.remove(os.path.join(MOUNT_DIR, file))
    vol_dataset.commit()
    return True


@stub.local_entrypoint()
def main():
    create_subsets.remote()
