# RetMask

Source code for the paper [**"From Interpretability to Performance: Optimizing Retrieval Heads for Long-Context Language Models"**](https://arxiv.org/abs/2601.11020) (ACL 2026 Findings).

## Trained Checkpoints

Trained checkpoints are available on Hugging Face:

| Model | Checkpoint |
|-------|-----------|
| Llama-3.1-8B-Instruct | [maym15/Llama-3.1-8B-Instruct-RetMask](https://huggingface.co/maym15/Llama-3.1-8B-Instruct-RetMask) |
| Qwen3-8B | [maym15/Qwen3-8B-RetMask](https://huggingface.co/maym15/Qwen3-8B-RetMask) |
| Olmo-3-7B-Instruct | [maym15/Olmo-3-7B-Instruct-RetMask](https://huggingface.co/maym15/Olmo-3-7B-Instruct-RetMask) |
| Olmo-3-7B-Think | [maym15/Olmo-3-7B-Think-RetMask](https://huggingface.co/maym15/Olmo-3-7B-Think-RetMask) |

## Environment

Tested on Python 3.12.2 and CUDA 12.8.0.

**Required packages:**
- `torch`
- `transformers`
- `flash-attn`
- `vllm`
- `trl`
- `deepspeed`
- `wandb`
- `rouge_score`

To set up the environment with `uv`:

```bash
uv sync
source .venv/bin/activate
```

## Pipeline Overview

RetMask consists of three steps:

1. **Retrieval Head Deactivation** — Detect retrieval heads and build an ablated model checkpoint.
2. **Contrastive Response Generation** — Sample responses from both the original and ablated models.
3. **Direct Preference Optimization** — Train the model to prefer original responses over ablated ones.

---

## Step 1: Retrieval Head Deactivation

This step builds on [Retrieval Head](https://github.com/nightdessert/Retrieval_Head/) [1], extended to support Llama-3(.1), Qwen3, and Olmo-3.

```bash
cd ./1_Retrieval_Head_Deactivation/
```

### Detection

Compute the retrieval score of each attention head:

```bash
python retrieval_head_detection.py --model_path $path_to_model --s 0 --e 5000
```

Outputs:
- NIAH task results → `results/graph/$basename_of_model`
- Attention head scores → `head_score/$basename_of_model`

### Deactivation

Build a checkpoint with retrieval heads masked:

```bash
python mask_retrieval_head.py \
    --model_name $path_to_model \
    --save_dir $path_to_save_ckpt \
    --mask_greater_than $threshold
```

This masks all heads with retrieval scores ≥ `threshold`.

**Variants:**

Mask the same number of randomly selected *non-retrieval* heads (prepend a minus sign):
```bash
python mask_retrieval_head.py \
    --model_name $path_to_model \
    --save_dir $path_to_save_ckpt \
    --mask_greater_than -$threshold
```

Mask the same number of randomly selected heads from *all* heads (add `--random`):
```bash
python mask_retrieval_head.py \
    --model_name $path_to_model \
    --save_dir $path_to_save_ckpt \
    --mask_greater_than -$threshold \
    --random
```

Mask the top-k heads by retrieval score:
```bash
python mask_retrieval_head.py \
    --model_name $path_to_model \
    --save_dir $path_to_save_ckpt \
    --mask_topk $topk
```

---

## Step 2: Contrastive Response Generation

This step uses [LMSYS-Chat-1M-Synth](https://huggingface.co/datasets/tokyotech-llm/lmsys-chat-1m-synth/tree/main) [2], which is derived from [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) [3]. Please agree to the respective licenses and obtain the datasets before proceeding.

> **Note:** Our experiments were conducted on LMSYS-Chat-1M-Synth. The code may work on LMSYS-Chat-1M directly, but this is not guaranteed.

```bash
cd ./2_Contrastive_Response_Generation/
```

Edit `--input_path` and `--output_path` in the appropriate shell script for your model:
- `exec_synthesize_assistant_response_using_llama.sh`
- `exec_synthesize_assistant_response_using_olmo.sh`
- `exec_synthesize_assistant_response_using_qwen.sh`

Then generate responses using `$path_to_model`:

```bash
bash set_up_vllm_api_llama_and_generate.sh $path_to_model $start $end
```

`$start` and `$end` control the sample index range, enabling parallel generation across multiple runs.

---

## Step 3: Direct Preference Optimization

```bash
cd ./3_Direct_Preference_Optimization/
```

Build DPO-formatted training data from the responses generated in Step 2:

```bash
python build_dpo_data.py \
    --chosen_path $path_to_chosen_data \
    --rejected_path $path_to_rejected_data \
    --output_path $path_to_output_data
```

Adjust `--output_dir` and other hyperparameters in the training scripts as needed.

> **Hardware:** Experiments were run on 4 × NVIDIA H100 GPUs. If using a different setup, adjust `configs/my_accelerate_config_zero1.yaml` and the relevant `run_dpo_*.sh` accordingly.

Launch training:

```bash
bash run_dpo_llama.sh $run_name $seed $path_to_training_data
```

Evaluate trained checkpoints using the [HELMET benchmark](https://github.com/princeton-nlp/helmet) [4].

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{ma2026interpretability,
    title     = "From Interpretability to Performance: Optimizing Retrieval Heads for Long-Context Language Models",
    author    = "Youmi Ma and Naoaki Okazaki",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2026",
    month     = jul,
    year      = "2026",
    address   = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    pages     = "(TBD)",
}
```

## References

[1] Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, and Yao Fu. 2025. Retrieval head mechanistically explains long-context factuality. In *ICLR*.

[2] Youmi Ma et al. 2025. Building Instruction-Tuning Datasets from Human-Written Instructions with Open-Weight Large Language Models. In *COLM*.

[3] Lianmin Zheng et al. 2024. LMSYS-Chat-1M: A large-scale real-world LLM conversation dataset. In *ICLR*.

[4] Howard Yen et al. 2025. HELMET: How to evaluate long-context models effectively and thoroughly. In *ICLR*.
