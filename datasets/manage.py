import asyncio

from pydantic import BaseModel

from modal_references import stub, vol_dataset, file_lock, file_metadata_queue
from datasets.upload import Metadata
from constants import MOUNT_DIR


class DatasetManage(BaseModel):
    type: str = "ls"
    file: str = None
    files: list[str] = None

    column: str = None

    splits: tuple[int, int, int, int] = None

@stub.function(volumes={MOUNT_DIR: vol_dataset}, concurrency_limit=1)
async def dataset_manage_wrapper(params: DatasetManage):
    ret = await dataset_manage_impl(params)
    return ret


async def dataset_manage_impl(params: DatasetManage):
    if params.type == "download":
        if not params.files:
            return False
        import os
        import zipfile
        from pathlib import Path
        file_location = os.path.join(MOUNT_DIR, "temp", "datasets.zip")
        os.makedirs(os.path.join(MOUNT_DIR, "temp"), exist_ok=True)
        with zipfile.ZipFile(file_location, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in params.files:
                archive.write(os.path.join(MOUNT_DIR, file), file)
                
        with open(file_location, "rb") as f:
            ret = f.read()
            import base64
            encoded_content = base64.b64encode(ret).decode("utf-8")
            return {"content": encoded_content, "filename": "datasets_" + "_".join([Path(path).stem for path in params.files]), "filetype": "zip"}
    elif params.type == "delete":
        if not params.files:
            return True
        vol_dataset.reload()
        import os
        for file in params.files:
            os.remove(os.path.join(MOUNT_DIR, file))
            # Make sure there's if statement to check deleted
            # Make sure that deleted instances throw an error
            file_metadata_queue.put(
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

        files: list[Metadata] = file_metadata_queue.get_many(9999, block=False)
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

        file_metadata_queue.put(
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
        file_metadata_queue.put_many(
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
