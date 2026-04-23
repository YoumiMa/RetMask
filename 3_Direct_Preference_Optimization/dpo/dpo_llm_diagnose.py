import torch
import logging
import itertools
import transformers
from typing import Optional
from dataclasses import dataclass
from transformers.trainer_utils import set_seed
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset, concatenate_datasets
from data_collator import DataCollatorForLastTurnOnlyLM

from transformers import (
    Trainer,
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    BitsAndBytesConfig,
)


logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_info()


        
def load_dpo_datasets(files, tokenizer):
    datasets = [] # multiple datasets
    
    for data_file in files:
        # Detect file format based on extension
        if data_file.endswith('.parquet'):
            dataset = load_dataset("parquet", data_files=data_file)
        elif data_file.endswith('.json') or data_file.endswith('.jsonl') or data_file.endswith('.jsonl.gz'):
            dataset = load_dataset("json", data_files=data_file)
        else:
            raise ValueError(f"Unsupported file format: {data_file}")
        
        dataset = dataset["train"]
        dataset = dataset.select(range(100))
        datasets.append(dataset)
    
    # Concatenate all datasets
    combined_dataset = concatenate_datasets(datasets)
    
    return combined_dataset

@dataclass
class DPOTrainingArguments:
    model_name_or_path: str
    data_files: list[str]
    eval_data_files: list[str] = None
    tokenizer_name_or_path: Optional[str] = None
    use_fast: bool = True
    additional_special_tokens: list[str] = None
    max_seq_length: int = 4096
    preprocessing_num_workers: int = 8
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention_2: bool = False
    use_peft: bool = False
    peft_target_model: Optional[str] = "llm-jp"
    peft_target_modules: Optional[list[str]] = None
    peft_lora_r: int = 8
    peft_lora_alpha: int = 32
    peft_lora_dropout: float = 0.05
    
    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("load_in_8bit and load_in_4bit are mutually exclusive")
        if self.peft_target_model and self.peft_target_modules is None:
            if self.peft_target_model == "llm-jp":
                self.peft_target_modules = ["c_attn", "c_proj", "c_fc"]
            elif self.peft_target_model == "llama":
                # https://github.com/serp-ai/LLaMA-8bit-LoRA/blob/main/finetune_peft_8bit.py
                self.peft_target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            elif self.peft_target_model == "llama-all":
                # https://note.com/kan_hatakeyama/n/ncd09c52d26c7
                self.peft_target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",
                    "embed_tokens",
                ]
            else:
                logger.warning(
                    f"peft_target_model '{self.peft_target_model}' is not supported, "
                    f"so peft_target_modules is set to None."
                )

    

def load_chat_datasets(files):
    datasets = [] # multiple datasets
    
    for data_file in files:

        dataset = load_dataset("json", data_files=data_file)
        dataset = dataset["train"]
        datasets.append(dataset)
    
    return concatenate_datasets(datasets)


def main():
    parser = HfArgumentParser((DPOConfig, DPOTrainingArguments))
    #print(parser)
    dpo_config, dpo_training_args = parser.parse_args_into_dataclasses()    
    dpo_config.dataset_num_proc = dpo_training_args.preprocessing_num_workers

    print(dpo_config)
    set_seed(dpo_config.seed)
    logger.info(f"Set seed: {dpo_config.seed}")
    
    tokenizer_name_or_path: str = (
        dpo_training_args.tokenizer_name_or_path or dpo_training_args.model_name_or_path
    )
    logger.info(f"Loading tokenizer from {tokenizer_name_or_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=dpo_training_args.use_fast,
        additional_special_tokens=dpo_training_args.additional_special_tokens,
        trust_remote_code=True,
    )

    print(tokenizer.special_tokens_map, "ids:", tokenizer.all_special_ids)
    logger.info("Loading data")

    dataset = load_dpo_datasets(dpo_training_args.data_files, tokenizer)
    print(dataset)

    for i, entry in enumerate(dataset):
        if not isinstance(entry["chosen"], str):
            print(f"Row {i} is problematic: {entry['chosen']}")
            break
        if not isinstance(entry["rejected"], str):
            print(f"Row {i} is problematic: {entry['chosen']}")
            break            
    exit(0)
    
    if dpo_training_args.eval_data_files:
        eval_dataset = load_chat_datasets(dpo_training_args.eval_data_files)
        dpo_config.eval_strategy = "steps"
    else:
        eval_dataset = None
    
    logger.info("Formatting prompts")

    logger.info(f"Loading model from {dpo_training_args.model_name_or_path}")
    
    logger.debug(
        f"AutoModelForCausalLM.from_pretrained({dpo_training_args.model_name_or_path}, trust_remote_code=True)"
    )
    model = AutoModelForCausalLM.from_pretrained(
        dpo_training_args.model_name_or_path,
        use_cache=True,
        trust_remote_code=True,
    )
    
    logger.info("Setting up trainer")
    trainer = DPOTrainer(
    model,
    ref_model=None,
    train_dataset=dataset,  # tokenized dataset
    args=dpo_config,  # training configs
    processing_class=tokenizer,  # tokenizer
)
    ##------------------debugging------------------
    # train_dataloader = trainer.get_train_dataloader()
    # first_batch = next(iter(train_dataloader))

    # print("\n" + "=" * 80)
    # print("First training batch:")
    # print(f"Batch keys: {first_batch.keys()}")
    
    # if 'input_ids' in first_batch:
    #     sample_ids = first_batch['input_ids'][0][:100]  # first 100 tokens
    #     print(f"\nSample input_ids: {sample_ids}")
    #     print(f"\nDecoded sample: {tokenizer.decode(sample_ids)}")
    
    # print("=" * 80 + "\n")

    ##------------------debugging------------------

    logger.info("Training")
    trainer.train(resume_from_checkpoint = dpo_config.resume_from_checkpoint)
    
    
    logger.info("Saving model")
    trainer.save_model()
    
    return


if __name__ == "__main__":
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
