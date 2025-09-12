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
| `--workers WORKERS` | Number of worker processes | `10` |
| `--prefetch PREFETCH` | Prefetch factor to read rows ahead of workers | `2` |
| `--min-dur MIN_DUR` | Minimum audio duration (sec) | `1.0` |
| `--max-dur MAX_DUR` | Maximum audio duration (sec) | `35.0` |
| `--vad-mode VAD_MODE` | Aggressiveness of VAD (0–3) | `2` |
| `--pause-sec PAUSE_SEC` | Minimum pause used for splitting | `0.3` |

### Examples
Download the `train` split of a public dataset:

```bash
python download_dataset.py nvidia/voice --split train --out data_voice
```

Download with custom limits:

```bash
python download_dataset.py bond005/taiga_speech_v2 --min-dur 1 --max-dur 15 \
    --workers 4 --out taiga_subset
```

## 2. Run Canary inference
`canary_inference.py` resolves dataset and output paths relative to the script
directory, so you can point to folders next to it or provide absolute paths.
Example:

```
python canary_inference.py --dataset-dir data_wav --out-dir predictions
```

The model cache is expected under `.hf/models--nvidia--canary-1b-v2`
relative to the script. To use a different Hugging Face cache or `.nemo`
file, edit the `MODEL_PATH` constant in `canary_inference.py`.

Use `--help` to see all available options. The script loads the Canary model and
writes predictions vs reference text for each audio file.

## 3. Analyse predictions
`dataset_analysis.py` resolves paths relative to the script directory and scans
the `predictions/` folder for subdirectories. Each subfolder is expected to
contain a `preds.jsonl` file with model outputs. For every dataset folder the
script writes analysis results to `analysis_output/<dataset_name>/`.

```
python dataset_analysis.py --preds-dir predictions --out-dir analysis_output \
    --semantic-weight 2.0 --wer-weight 1.0
```

For each dataset the script computes WER, SER and semantic similarity for each
utterance and derives a combined **difficulty** score, with semantic mismatch
emphasised over raw WER:

```
difficulty = 0.3 * WER + 0.7 * (1 - semantic_similarity)
```

The script splits the dataset into 30% easy and 70% difficult examples (after
trimming difficulty outliers) and builds a Pareto front over WER vs semantic
similarity. Distribution plots and the WER/semantic Pareto diagram are written
alongside the front and dominated examples in `pareto_front.jsonl` and
`beyond_pareto.jsonl`. Use `--help` to see all options, including the
`--tail-fraction` parameter that controls outlier trimming.

The script depends on `sentence-transformers` and `matplotlib` which can be
installed via pip:

```
pip install sentence-transformers matplotlib
```
