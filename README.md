# STT Dataset Preparation (Ru)

Scripts extracted from the original notebook to prepare Russian speech datasets for NVIDIA Canary.

## 1. Download dataset
```
python download_dataset.py <dataset_repo> --split <split> --out <out_dir> [--hf-token YOUR_TOKEN]
```
The script downloads a HuggingFace dataset split, filters Russian samples, converts audio to 16kHz mono WAV and writes a `manifest.jsonl` file.

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
