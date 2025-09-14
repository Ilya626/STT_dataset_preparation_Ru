# Runpod: Canary LoRA (Native NeMo) — Quick Start

All training scripts live under `/workspace/training`. Run commands from the
repository root so the `training/` prefix resolves correctly.

## Start training (auto-download .nemo)

```bash
python training/runpod_nemo_canary_lora.py \
  --auto_download --model_id nvidia/canary-1b-v2 \
  --nemo /workspace/models/canary-1b-v2.nemo \
  --train /workspace/data/train_portable.jsonl \
  --val   /workspace/data/val_portable.jsonl \
  --outdir /workspace/exp/canary_ru_lora_a6000 \
  --export /workspace/models/canary-ru-lora-a6000.nemo \
  --bs 8 --accum 1 --precision bf16 --num_workers 8 \
  --adapter_name ru_lora --enc_lora_layers 6 --dec_lora_layers 2 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05
```

Notes
- If VRAM allows on A6000 (48 GB), try `--bs 12`.
- Resume training from last checkpoint:

```bash
python training/runpod_nemo_canary_lora.py \
  --nemo /workspace/models/canary-1b-v2.nemo \
  --train /workspace/data/train_portable.jsonl \
  --val   /workspace/data/val_portable.jsonl \
  --outdir /workspace/exp/canary_ru_lora_a6000 \
  --export /workspace/models/canary-ru-lora-a6000.nemo \
  --bs 8 --accum 1 --precision bf16 --num_workers 8 \
  --adapter_name ru_lora --enc_lora_layers 6 --dec_lora_layers 2 \
  --resume /workspace/exp/canary_ru_lora_a6000/last.ckpt
```

Where
- Outputs/checkpoints: `/workspace/exp/canary_ru_lora_a6000/`
- Exported merged `.nemo`: `/workspace/models/canary-ru-lora-a6000.nemo`
- Caches: `.hf`, `.torch`, temps: `.tmp` (in repo root)

## Unified launcher (download ZIP + finetune)

```bash
python training/runpod_canary_launcher.py \
  --dataset_url https://huggingface.co/USERNAME/REPO/resolve/main/fine-tuning-dataset.zip \
  --method lora \
  --nemo /workspace/models/canary-1b-v2.nemo \
  --outdir /workspace/exp/canary_ru_lora \
  --export /workspace/models/canary-ru-lora.nemo \
  --preset a6000-fast \
  --adapter_name ru_lora --enc_lora_layers 6 --dec_lora_layers 2
```

Or using an already unpacked folder:

```bash
python training/runpod_canary_launcher.py \
  --dataset_dir /workspace/data/fine-tuning-dataset-YYYYMMDD \
  --method partial \
  --nemo /workspace/models/canary-1b-v2.nemo \
  --outdir /workspace/exp/canary_partial \
  --export /workspace/models/canary-partial.nemo \
  --unfreeze_encoder_last 4 --unfreeze_decoder_last 2 --unfreeze_head \
  --preset a6000-fast
```
