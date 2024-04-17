import os
from pydantic import BaseModel
from fastapi import UploadFile, File
import constants
from modal_references import stub, vol_dataset

# Metadata is an object that keeps track of file metadata while in a queue to be processed.
class Metadata(BaseModel):
    lines: int
    size: int
    columns: list[str]
    filtered: bool
    file: str
    deleted: bool
    original_file: str
    split_type: str

@stub.function(volumes={constants.MOUNT_DIR: vol_dataset})
async def dataset_upload_impl(content: bytes, filename: str):
    import pandas as pd
    import numpy as np
    from io import StringIO

    # 1. Save the file
    data = content.decode("utf-8")
    df = pd.read_csv(StringIO(data), low_memory=False)
    df = df.fillna(np.nan)
    for col in constants.COLUMNS:
        if col in df.columns:
            df = df.dropna(subset=[col])
    df.to_csv(os.path.join(constants.MOUNT_DIR, filename), index=False)

    # Add metadata to be saved into a queue. Will call on page load.
    info = Metadata(
        split_type="Raw",
        lines=len(df),
        size=os.path.getsize(os.path.join(constants.MOUNT_DIR, filename)),
        columns=list(df.columns),
        filtered=False,
        file=filename,
        deleted=False,
        original_file=filename,
    )
    stub.file_metadata_queue.put(info)

    # 2. Commit
    import asyncio
    asyncio.create_task(vol_dataset.commit.aio())

    # 3. Show preview
    return info
