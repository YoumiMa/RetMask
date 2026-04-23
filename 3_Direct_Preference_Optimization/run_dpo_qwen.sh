LR=5e-7
MINLR=5e-8
WD=0.1
NAME=$1; shift
SEED=$1; shift
DATA=("$@")
NAME=Qwen3-${NAME}_LR_${LR}_MINLR_${MINLR}_WD_${WD}

accelerate launch --config_file configs/my_accelerate_config_zero2.yaml dpo/dpo_llm.py --output_dir /path/to/output/dir/${NAME}_${SEED} \
--run_name $NAME \
--data_files ${DATA[*]} \
--model_name_or_path Qwen/Qwen3-8B \
--tokenizer_name_or_path Qwen/Qwen3-8B \
--bf16 true \
--num_train_epochs 2 \
--per_device_train_batch 1 \
--gradient_accumulation_steps 128 \
--gradient_checkpointing \
--optim adamw_torch \
--adam_beta2 0.95 \
--learning_rate ${LR} \
--lr_scheduler_type cosine_with_min_lr \
--lr_scheduler_kwargs "{\"min_lr\":${MINLR}}" \
--weight_decay ${WD} \
--warmup_ratio 0.1 \
--logging_steps 10 \
--save_steps 500 \
--seed ${SEED} \
--report_to wandb
