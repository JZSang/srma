import modal

### Raw dataset storage
vol_dataset = modal.Volume.persisted("srma-dataset")

### Fewshot GPTCoT storage
cot_vol = modal.Volume.persisted("srma-cot")

### Storage of test results
vol_save_results = modal.Volume.persisted("srma-results")

### MODAL INITIALIZERS
stub = modal.Stub(
    "srma-retool",
    image=modal.Image.debian_slim().pip_install(
        ["google-generativeai", "pandas", "openai", "tiktoken", "mistralai", "mistral-common", "fastapi==0.100.0"]
    ),
)
### Producer-Consumer Queues
stub.result_queue = modal.Queue.new()
stub.cot_result_queue = modal.Queue.new()

### Process status tracker
stub.status_tracker = modal.Dict.new()

stub.file_lock = modal.Dict.new()
stub.file_metadata_queue = modal.Queue.new()