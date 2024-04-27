import modal

### Raw dataset storage
vol_dataset = modal.Volume.from_name("srma-dataset")

### Fewshot GPTCoT storage
cot_vol = modal.Volume.from_name("srma-cot")

### Storage of test results
vol_save_results = modal.Volume.from_name("srma-results")

### Storage of intermediate batches
vol_save_intermediate_batches = modal.Volume.from_name("srma-batches-intermediate")

### MODAL INITIALIZERS
stub = modal.Stub(
    "srma-retool",
    image=modal.Image.debian_slim().pip_install(
        [
            "google-generativeai",
            "pandas",
            "openai==1.23.2",
            "tiktoken",
            "mistralai",
            "mistral-common",
            "fastapi==0.100.0",
            "pymongo[srv]==3.11"
        ]
    ),
)

### Producer-Consumer Queues
result_queue = modal.Queue.from_name("result_queue", create_if_missing=True)
cot_result_queue = modal.Queue.from_name("cot_result_queue", create_if_missing=True)

### Process status tracker
status_tracker_global_dictionary = modal.Dict.from_name("status_tracker", create_if_missing=True)

file_lock = modal.Dict.from_name("file_lock", create_if_missing=True)
file_metadata_queue = modal.Queue.from_name(
    "file_metadata_queue", create_if_missing=True
)
