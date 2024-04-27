import dataclasses
import modal
from openai import files
from pydantic import BaseModel
from constants import MOUNT_DIR
from modal_references import stub, vol_save_intermediate_batches
from models.openai import extract_from_response_openai
from mongo.mongo import new_mongo_client
from tasks.abstract_resource import AbstractRunResource
from tasks.evaluation import is_excluded
from tasks.run_test import Item, finalize_results
from tasks.files import filepath_from_unique_id, load_final_results, save_final_results

class LoadBatches(BaseModel):
    pass

class LoadBatch(BaseModel):
    ensemble_id: str

class CheckAndUpdateBatches(BaseModel):
    pass

@stub.function(secrets=[modal.Secret.from_name("mongo-db-atlas-srma"), modal.Secret.from_name("srma-openai")], volumes={MOUNT_DIR: vol_save_intermediate_batches},)
async def check_and_update_batches_impl(_: CheckAndUpdateBatches):
    from pymongo import UpdateOne
    import asyncio
    import json
    import time
    from openai import AsyncOpenAI
    mongo_client = new_mongo_client()
    client = AsyncOpenAI()
    
    # OpenAI Batch API response mirror collection
    individual_collection = mongo_client.get_database("SRMA").get_collection("individual_batch_status")
    # SRMA run ensembles
    collection = mongo_client.get_database("SRMA").get_collection("run_batched")
    
    # Find all those that have not completed the retrieval process
    results = collection.find({"status": {"$not": {"$regex": ".*srma_retrieved.*"}}})
    
    completed_status = "completed"
    failure_statuses = ["failed", "expired", "cancelling", "cancelled"]
    finished_statuses = [completed_status] + failure_statuses
    
    # For each that is incomplete
    for document in results:
        ensemble_id = document["ensemble_id"]
        batches = document["batches"]
        status = document["status"]
        
        # Update Batch status from OpenAI
        bulk_write = []
        list_of_new_batches = await asyncio.gather(*[client.batches.retrieve(batch_id) for batch_id in batches])
        for new_batch in list_of_new_batches:
            bulk_write.append(UpdateOne({"id": new_batch.id}, {"$set": new_batch.to_dict()}))
        individual_collection.bulk_write(bulk_write)
        
        # 1. Immediately check for failures. If there is a failure in the batch set, we discard it.
        if set(map(lambda x: x.status, list_of_new_batches)).intersection(failure_statuses):
            status = "srma_retrieved_failed"
            collection.update_one({"_id": document["_id"]}, {"$set": {"status": status, "completed_at": int(time.time())}}) 
            print(f"Batch {ensemble_id} failed. Status: {status}")
            continue
        
        # 2. If all batches are complete
        if set(map(lambda x: x.status, list_of_new_batches)) == {completed_status} and len(list_of_new_batches) == len(batches):
            print(f"{ensemble_id}: Batch is complete on OpenAI's side. Moving to processing.")
            result_queue = []
            
            # 1. Save input parameters
            item = Item(**document["item"])
            print(f"{ensemble_id}: Input params saved. {item.model_dump()}")
            
            new_batches = list(map(lambda x: (x.output_file_id, x.id), list_of_new_batches))
            latest_completed_time = max(map(lambda x: x.created_at, list_of_new_batches))
            
            # 2. Read batch results and collect jsons into a list.
            #    An element of the list contains a single generation_task list
            #    An element of the generation_task list is a single run
            files = await asyncio.gather(*[client.files.content(output_file_id) for output_file_id, _ in new_batches])
            files = list(map(lambda x: [json.loads(line) for line in x.iter_lines()], files))
            # files is a list of generation tasks. Each line in the generation task is a response from the API.
            
            for i, generation_task in enumerate(files):
                _, batch_id = new_batches[i]
                print(f"{ensemble_id}: Processing individual batch {batch_id}")
                # 3. We saved the intermediate values for a single generation task result queue, so we load it up
                with open(f"{MOUNT_DIR}/intermediate_{batch_id}.json", "r") as f:
                    import json
                    raw_result_queue = json.load(f)
                raw_result_dict = {}
                for raw_result_element in raw_result_queue:
                    raw_result_dict[raw_result_element["abstract_id"]] = AbstractRunResource(**raw_result_element)
                # 4. We match the setup generation task with the new results from OpenAI
                for line in generation_task:
                    abstract_id = line["custom_id"]
                    raw_result: AbstractRunResource = raw_result_dict[abstract_id]
                    error = None
                    try:
                        if line["error"]:
                            error = line["error"]
                        llm_answer = extract_from_response_openai(line["response"]["body"], is_json=True)
                        predicted_value = is_excluded(llm_answer, raw_result.test_abstract)
                    except (Exception) as e:
                        error = e
                    if error:
                        raw_result.finalize(
                            skipped=True,
                            correct=False,
                            predicted_value="fail",
                            llm_answer="fail",
                            error=error,
                        )
                    else:
                        raw_result.finalize(
                            skipped=False,
                            correct=predicted_value == raw_result.actual_value,
                            predicted_value=predicted_value,
                            llm_answer=llm_answer,
                            error=None,
                        )
                    result_queue.append(raw_result.__dict__)
            print(f"{ensemble_id}: {len(result_queue)} total flattened results processed.")
            # 5. Our result_queue now matches up with the state of a normal synchronous run. We throw them into finalize_results
            #    to collate them into one "ensemblised" bunch of results
            finalized_results = finalize_results(result_queue, item, ensemble_threshold=document["ensemble_threshold"])
            
            print(f"{ensemble_id}: Finalized results {finalized_results.total_correct}/{finalized_results.total} correct.")
            ret = {
                # Overlay inputs
                **item.model_dump(),
                
                # Overlay finalized results (may overwrite some things in item, that's ok)
                **dataclasses.asdict(finalized_results),
            }
            
            save_final_results.spawn(ret, unique_id=ensemble_id)
            
            # Now we omit results and update the mongo (results are huge so we only save it in our storage)
            ret["results"] = None
            collection.update_one({"_id": document["_id"]}, {"$set": {"status": "srma_retrieved", "run_results": ret, "data_file_name": filepath_from_unique_id(ensemble_id), "completed_at": int(latest_completed_time)}})
        else:
            # find status of the first batch that is not completed
            incomplete_status = list(filter(lambda x: x.status not in finished_statuses, list_of_new_batches))[0].status
            collection.update_one({"_id": document["_id"]}, {"$set": {"status":incomplete_status}})
                        
@stub.function(secrets=[modal.Secret.from_name("mongo-db-atlas-srma")])
def load_batches(_: LoadBatches):
    mongo_client = new_mongo_client()
    
    check_and_update_batches_impl.remote(CheckAndUpdateBatches())
    
    collection = mongo_client.get_database("SRMA").get_collection("run_batched")
    # get everything ordered by the time it was created (descending)
    results = collection.find({}, {"_id": 0}).sort("created_at", -1)
    
    return list(results)
    
@stub.function()
def load_batch(params: LoadBatch):
    return load_final_results.local(unique_id=params.ensemble_id)