#!/bin/bash

MODEL_NAME_OR_PATH=$1
START=$2
END=$3
URL=$4

python synthesize_assistant_response.py \
--input_path "/path/to/input/file" \
--output_path "/path/to/output/dir/lmsys-chat-1m_synthesized_English_llama_${START}_${END}.jsonl.gz" \
--api_key "EMPTY" \
--api_base_url ${URL} \
--target_language="English" \
--model ${MODEL_NAME_OR_PATH} \
--n 1 \
--max_retries 3 \
--temperature 0.6 \
--top_p 1.0 \
--frequency_penalty 0.0 \
--max_tokens 2048 \
--num_threads 200 \
--start_index ${START} \
--end_index ${END} \
