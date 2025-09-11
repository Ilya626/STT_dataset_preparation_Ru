# STT Dataset Preparation (Ru)

Scripts extracted from the original notebook to prepare Russian speech datasets for NVIDIA Canary.

## 1. Download dataset
```
python download_dataset.py <dataset_repo> --split <split> --out <out_dir> [--hf-token YOUR_TOKEN]
```
The script downloads a HuggingFace dataset split, filters Russian samples, converts audio to 16kHz mono WAV and writes a `manifest.jsonl` file.

### Options
`download_dataset.py` accepts a number of flags to control processing:

| Option | Description | Default |
| --- | --- | --- |
| `--split SPLIT` | Dataset split to download | `train` |
| `--out OUT` | Output directory | `data_wav` |
| `--hf-token HF_TOKEN` | Hugging Face access token | env `HF_TOKEN` |
| `--lang-regex LANG_REGEX` | Regex to match Russian samples | `(^|[-_])ru([-_]|$)|russian` |
| `--cache-dir CACHE_DIR` | Datasets cache directory | `hf_cache` |
| `--workers WORKERS` | Number of parallel processes | `10` |
| `--min-dur MIN_DUR` | Minimum audio duration (sec) | `1.0` |
| `--max-dur MAX_DUR` | Maximum audio duration (sec) | `35.0` |
| `--vad-mode VAD_MODE` | Aggressiveness of VAD (0–3) | `2` |
| `--pause-sec PAUSE_SEC` | Minimum pause used for splitting | `0.3` |
| `--chunk-size CHUNK_SIZE` | Rows processed per batch | `100` |

### Examples
Download the `train` split of a public dataset:

```bash
python download_dataset.py nvidia/voice --split train --out data_voice
```

Download with custom limits and batching:

```bash
python download_dataset.py bond005/taiga_speech_v2 --min-dur 1 --max-dur 15 \
    --chunk-size 200 --workers 4 --out taiga_subset
```

## 2. Run Canary inference
Edit the path constants at the top of `canary_inference.py` to set the dataset
directory, model and output location, then run:

```
python canary_inference.py
```
The script loads the Canary model and writes predictions vs reference text for each audio file.

## 3. Analyse predictions
Edit the path constants at the top of `dataset_analysis.py` to select the
predictions file and output directory, then run:

```
python dataset_analysis.py
```
The script computes WER, SER and semantic similarity for each utterance, splits
the dataset into 30% easy and 70% difficult examples (after trimming outliers)
and produces distribution plots.

The script depends on `sentence-transformers` and `matplotlib` which can be
installed via pip:

```
pip install sentence-transformers matplotlib
```
