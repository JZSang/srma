### LLM SETTINGS
DEFAULT_SECONDS_PER_REQUEST = 0.007
MAX_COMPLETION_TOKENS = 100000
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
    "gpt-4o-2024-05-13": {
        "max_requests_per_minute": 10000,
        "max_tokens_per_minute": 1500000,
    },
    "gpt-4o-2024-08-06": {
        "max_requests_per_minute": 10000,
        "max_tokens_per_minute": 1500000,
    },
    "gpt-4o-2024-11-20": {
        "max_requests_per_minute": 10000,
        "max_tokens_per_minute": 1500000,
    },
    "gpt-3.5-turbo-0125": {
        "max_requests_per_minute": 10000,
        "max_tokens_per_minute": 2000000,
    },
    "gemini-pro": {
        "max_requests_per_minute": 360,
        "max_tokens_per_minute": float("inf"),
    },
    "gemini-1.5-flash-latest": {
        "max_requests_per_minute": 15,
        "max_tokens_per_minute": 1000000,
        "max_requests_per_second": 2,
    },
    "open-mixtral-8x22b-2404": {
        "max_requests_per_minute": 300,
        "max_tokens_per_minute": 2000000,
        "max_requests_per_second": 5,
    },
    "mistral-large-2402": {
        "max_requests_per_minute": 300,
        "max_tokens_per_minute": 2000000,
        "max_requests_per_second": 5,
    },
}
TOKENIZER = "cl100k_base"
TEXT_SEED = 1337

### MODAL SETTINGS
WORKSPACE = "jzsang"

### General dataset mounting
MOUNT_DIR = "/data"
SAVE_BATCH_MOUNT_DIR = "/batches"

### GPTCoT storage
MOUNT_DIR_COT = "/cot"

### Input specific info
COLUMNS = ["Abstract", "Content"]
