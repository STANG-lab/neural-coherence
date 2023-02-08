"""
Defines core functions for downloading and transforming the Switchboard dataset.

"""
import os
import datasets as ds
from pathlib import Path
import polars as pl
from typing import Dict

DATA_PATH = Path("../data/switchboard")


def load_dataset_raw(raw_data_dir: Path = DATA_PATH / "raw") -> Dict[str, pl.DataFrame]:
    """Load the raw Switchboard dataset from disk if it exists, or download it if not."""
    data = dict([(f.stem, __load_from_disk(f)) for f in raw_data_dir.glob("*.parquet")])
    return data or __dl_dataset(raw_data_dir)


def load_dataset_grouped(
    raw_dir: Path = DATA_PATH / "raw",
    grouped_dir: Path = DATA_PATH / "grouped",
) -> Dict[str, pl.DataFrame]:
    """Load the cleaned and grouped by utterance Switchboard dataset from disk
    if it exists, or build it if not."""
    if not grouped_dir.exists():
        os.makedirs(grouped_dir)

    data = dict([(f.stem, __load_from_disk(f)) for f in grouped_dir.glob("*.parquet")])

    return data or dict(
        [
            (name, __group_dataset(name, df, grouped_dir))
            for name, df in load_dataset_raw(raw_dir).items()
        ]
    )


###### PRIVATE ######


def __load_from_disk(data_path: Path) -> pl.DataFrame:
    return pl.read_parquet(data_path)


def __dl_dataset(data_dir: Path) -> Dict[str, pl.DataFrame]:
    """Download the Switchboard dataset from huggingface and save it in Parquet format."""
    dataset: ds.DatasetDict = ds.load_dataset("swda", data_dir=data_dir)  # type: ignore

    if not data_dir.exists():
        os.makedirs(data_dir)

    for sl in dataset.keys():
        dataset[sl].to_parquet(str(data_dir / sl) + ".parquet")

    return load_dataset_raw(data_dir)


def __group_dataset(
    sl_name: str, raw_df: pl.DataFrame, grouped_data_dir: Path
) -> pl.DataFrame:
    def cast_cols(df: pl.LazyFrame):
        """Select only the cols we need and cast them to efficient datatypes."""
        return df.select(
            [
                pl.col("text"),
                pl.col("subutterance_index").cast(pl.UInt8),
                pl.col("conversation_no").cast(pl.UInt16),
                pl.col("utterance_index").cast(pl.UInt16),
                pl.col("prompt").cast(pl.Categorical),
                pl.col("caller").cast(pl.Categorical),
            ]
        )

    def group_utters(df: pl.LazyFrame):
        """Group consecutive utterances from the same speaker into a single row."""
        return (
            df.with_columns(
                pl.col("text")
                .str.concat(" ")
                .over(["conversation_no", "utterance_index"])
            )
            .filter(pl.col("subutterance_index") == 1)
            .drop("subutterance_index")
        )

    def clean_text(df: pl.LazyFrame):
        """Remove switchboard annotations from the text column."""
        return df.with_columns(
            pl.col("text")
            .str.replace_all(r"(\{\w*)|( })|(\[ )|( \+)|( \])|( /)|(<>)", "")
            .str.replace_all(r"<(L|l)aughter>", "Haha")
            .str.strip()
        )

    df = raw_df.lazy().pipe(cast_cols).pipe(group_utters).pipe(clean_text).collect()
    df.write_parquet(str(grouped_data_dir / sl_name) + ".parquet")
    return df
