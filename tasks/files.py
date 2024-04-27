from constants import SAVE_BATCH_MOUNT_DIR
from modal_references import stub, vol_save_results

def filename_from_unique_id(unique_id):
    return f"results_{unique_id}.json"

def filepath_from_unique_id(unique_id):
    import os
    return os.path.join(SAVE_BATCH_MOUNT_DIR, filename_from_unique_id(unique_id))

@stub.function(volumes={SAVE_BATCH_MOUNT_DIR: vol_save_results})
async def save_final_results(final_results, unique_id):
    import json
    final_results["results"] = [result.dict() for result in final_results["results"]]
    with open(filepath_from_unique_id(unique_id), "w") as f:
        json.dump(final_results, f)
    vol_save_results.commit()
    
@stub.function()
def load_final_results(unique_id):
    import json
    import os
    filepath = filepath_from_unique_id(unique_id)
    if not os.path.exists(filepath):
        vol_save_results.reload()
    with open(filepath, "r") as f:
        loaded = json.load(f)
        return loaded