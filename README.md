# STT Dataset Preparation (Ru)

Scripts extracted from the original notebook to prepare Russian speech datasets for NVIDIA Canary.

## 1. Download dataset
```
python download_dataset.py <dataset_repo> --split <split> --out <out_dir> [--hf-token YOUR_TOKEN]
```
The script downloads a HuggingFace dataset split, filters Russian samples, converts audio to 16kHz mono WAV and writes a `manifest.jsonl` file.

## 2. Run Canary inference
```
python canary_inference.py <manifest.jsonl> --out preds.jsonl
```
Loads the Canary model and writes predictions vs reference text for each audio file.

## 3. Analyse predictions
```
python dataset_analysis.py preds.jsonl
```
Prints word error rate (WER) and sentence error rate (SER) for the predictions file.
