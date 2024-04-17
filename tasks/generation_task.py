import constants

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
            frac=1, random_state=1
        )
        self.included_abstracts = included_abstracts.sample(
            frac=1, random_state=1
        )
        self.get_next_abstract_excluded = self.get_next_abstract_generator(True)
        self.get_next_abstract_included = self.get_next_abstract_generator(False)

    # generator
    def get_next_abstract_generator(self, exclude):
        if exclude:
            for i,abstract in enumerate(self.excluded_abstracts):
                yield abstract, "excluded_" + str(i)
        elif not exclude:
            for i,abstract in enumerate(self.included_abstracts):
                yield abstract, "included_" + str(i)
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
                os.path.join(constants.MOUNT_DIR_COT, f"cot_{self.gpt_cot_id}.csv")
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
        import numpy as np
        self.seed_generator = np.random.default_rng(seed)

    def new_default_rng(self):
        """
        Used to ensure generation is consistent across runs for subtree operations
        """
        import numpy as np
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
