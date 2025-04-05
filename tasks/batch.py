import dataclasses
import modal
import asyncio
import json
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

class BatchManagement():
    from anthropic.types.beta.messages.beta_message_batch import BetaMessageBatch
    from openai.types.batch import Batch
        
    # abstract function
    async def retrieve_batch(self, batch_id: str) -> Batch | BetaMessageBatch:
        pass
    
    def has_failed(self, batches: list[Batch | BetaMessageBatch]) -> bool:
        pass
    
    def has_succeeded(self, batches: list[Batch | BetaMessageBatch]) -> bool:
        pass
    
    def get_statuses(self, batches: list[Batch | BetaMessageBatch]) -> list[str]:
        pass    
    
    def get_batches_with_ids(self, batches: list[Batch | BetaMessageBatch]) -> list[tuple[str, str]]:
        pass
    
    def get_latest_completed_time(self, batches: list[Batch | BetaMessageBatch]) -> int:
        pass
    
    async def download_batch_results(self, batches: list[Batch | BetaMessageBatch]) -> list[list[any]]:
        pass
    
    def get_error_from_line(self, line: dict) -> str | None:
        pass
    
    def extract_from_response(self, response):
        pass
        
class OpenAIBatchManagement(BatchManagement):
    from openai.types.batch import Batch
    completed_status = "completed"
    failure_statuses = ["failed", "expired", "cancelling", "cancelled"]
    finished_statuses = [completed_status] + failure_statuses
    
    def __init__(self):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI()
        self.name = "OpenAI"
    async def retrieve_batch(self, batch_id: str):
        return await self.client.batches.retrieve(batch_id)
    
    def has_failed(self, batches: list[Batch]) -> bool:
        return set(map(lambda x: x.status, batches)).intersection(self.failure_statuses)
    
    def has_succeeded(self, batches: list[Batch]) -> bool:
        return set(map(lambda x: x.status, batches)) == {self.completed_status}
    
    def get_statuses(self, batches: list[Batch]) -> list[str]:
        return list(map(lambda x: x.status, batches))
    
    def get_batches_with_ids(self, batches: list[Batch]) -> list[tuple[str, str]]:
        return list(map(lambda x: (x.output_file_id, x.id), batches))
    
    def get_latest_completed_time(self, batches: list[Batch]) -> int:
        return max(map(lambda x: x.completed_at, batches))
    
    async def download_batch_results(self, batches: list[Batch]) -> list[list[any]]:
        files = await asyncio.gather(*[self.client.files.content(batch.output_file_id) for batch in batches if batch.output_file_id])
        files = list(map(lambda x: [json.loads(line) for line in x.iter_lines()], files))
        return files
    
    def get_error_from_line(self, line: dict) -> str | None:
        return line["error"]
    
    def extract_from_response(self, response):
        return extract_from_response_openai(response["response"]["body"], is_json=True)
    
class AnthropicBatchManagement(BatchManagement):
    from anthropic.types.beta.messages.beta_message_batch import BetaMessageBatch
    completed_status = "ended"
    failure_statuses = ["errored", "canceled", "expired", "canceling"]
    finished_statuses = [completed_status] + failure_statuses
    
    def __init__(self):
        from anthropic import Anthropic
        self.client = Anthropic()
        self.name = "Anthropic"
    
    async def retrieve_batch(self, batch_id: str):
        return self.client.beta.messages.batches.retrieve(batch_id)
    
    def has_failed(self, batches: list[BetaMessageBatch]) -> bool:
        return set(map(lambda x: x.processing_status, batches)).intersection(self.failure_statuses)
    
    def has_succeeded(self, batches: list[BetaMessageBatch]) -> bool:
        return set(map(lambda x: x.processing_status, batches)) == {self.completed_status}
    
    def get_statuses(self, batches: list[BetaMessageBatch]) -> list[str]:
        return list(map(lambda x: x.processing_status, batches))
    
    def get_batches_with_ids(self, batches: list[BetaMessageBatch]) -> list[tuple[str, str]]:
        return list(map(lambda x: (x.id, x.id), batches))
    
    def get_latest_completed_time(self, batches: list[BetaMessageBatch]) -> int:
        return max(map(lambda x: int(x.ended_at.timestamp()), batches))
    
    async def download_batch_results(self, batches: list[BetaMessageBatch]) -> list[list[any]]:
        files = [[line.to_dict(mode="json") for line in self.client.beta.messages.batches.results(batch.id)] for batch in batches]
        return files
    
    def get_error_from_line(self, line: dict) -> str | None:
        if line["result"]["type"] == "errored":
            return line["result"]["error"]
        return None

    def extract_from_response(self, response):
        for content_block in response["result"]["message"]["content"]:
            if "type" in content_block:
                if content_block["type"] == "text" and "text" in content_block:
                    return content_block["text"]
            elif "text" in content_block:
                return content_block["text"]
            

@stub.function(secrets=[modal.Secret.from_name("mongo-db-atlas-srma"), modal.Secret.from_name("srma-openai"), modal.Secret.from_name("anthropic-secret")], volumes={MOUNT_DIR: vol_save_intermediate_batches},)
async def check_and_update_batches_impl(_: CheckAndUpdateBatches):
    from pymongo import UpdateOne
    import asyncio
    import json
    import time
    mongo_client = new_mongo_client()
    
    apis: list[BatchManagement] = [OpenAIBatchManagement(), AnthropicBatchManagement()]

    # OpenAI Batch API response mirror collection
    individual_collection = mongo_client.get_database("SRMA").get_collection("individual_batch_status")
    # SRMA run ensembles
    collection = mongo_client.get_database("SRMA").get_collection("run_batched")
    
    # Find all those that have not completed the retrieval process
    results = collection.find({"status": {"$not": {"$regex": ".*srma_retrieved.*"}}})
    

    # For each that is incomplete
    for document in results:
        ensemble_id = document["ensemble_id"]
        batches = document["batches"]
        status = document["status"]
        
        if "msgbatch" in batches[0]:
            api = AnthropicBatchManagement()
        else:
            api = OpenAIBatchManagement()
        
        # Update Batch status from OpenAI
        bulk_write = []
        list_of_new_batches = await asyncio.gather(*[api.retrieve_batch(batch_id) for batch_id in batches])
        for new_batch in list_of_new_batches:
            bulk_write.append(UpdateOne({"id": new_batch.id}, {"$set": new_batch.to_dict()}))
        individual_collection.bulk_write(bulk_write)
        
        # 1. Immediately check for failures. If there is a failure in the batch set, we discard it.
        # if set(map(lambda x: x.status, list_of_new_batches)).intersection(failure_statuses):
        if api.has_failed(list_of_new_batches):
            status = "srma_retrieved_failed"
            collection.update_one({"_id": document["_id"]}, {"$set": {"status": status, "completed_at": int(time.time())}}) 
            print(f"Batch {ensemble_id} failed. Status: {status}")
            continue
        
        # 2. If all batches are complete
        if api.has_succeeded(list_of_new_batches) and len(list_of_new_batches) == len(batches):
            print(f"{ensemble_id}: Batch is complete on {api.name}'s side. Moving to processing.")
            result_queue = []
            
            # 1. Save input parameters
            item = Item(**document["item"])
            print(f"{ensemble_id}: Input params saved. {item.model_dump()}")
            
            new_batches = api.get_batches_with_ids(list_of_new_batches)
            latest_completed_time = api.get_latest_completed_time(list_of_new_batches)
            
            # 2. Read batch results and collect jsons into a list.
            #    An element of the list contains a single generation_task list
            #    An element of the generation_task list is a single run
            files = await api.download_batch_results(list_of_new_batches)
            # files is a list of generation tasks. Each line in the generation task is a response from the API.
            try:
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
                            # insert model
                            if api.get_error_from_line(line):
                                error = api.get_error_from_line(line)
                            else:
                                llm_answer = api.extract_from_response(line)
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
            except Exception as e:
                print(f"{ensemble_id}: Error processing batch {batch_id}: {e}")
                continue
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
            incomplete_status = api.get_statuses(list_of_new_batches)[0]
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