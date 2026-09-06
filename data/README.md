# Avazu data

The project uses the
[`EvgeniaKyriazi/ctr-prediction-dataset`](https://huggingface.co/datasets/EvgeniaKyriazi/ctr-prediction-dataset)
version of the Avazu click-through-rate dataset. The source currently provides an
800,000-row `train` split and a 200,000-row `test` split in Parquet format.

## Acquire the data

Dataset acquisition is separate from the core project dependencies because the
experiment code only needs the resulting Parquet files.

```powershell
python -m pip install datasets jupyter
jupyter notebook load_data.ipynb
```

Run both notebook cells. They create:

```text
data/raw/avazu_train.parquet
data/raw/avazu_test.parquet
```

The files occupy roughly 40 MB and are ignored by Git.

## Important split detail

The Hugging Face `train` and `test` splits are random source partitions: both cover
the full date range and both contain labels. They are not used as the project's
experimental train/test split.

The loader concatenates them, validates their schemas, performs a stable sort by
`(datetime, __index_level_0__)`, and only then creates chronological model splits.
This prevents future timestamps from appearing in the preprocessing prefix.
