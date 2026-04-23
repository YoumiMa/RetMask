"""
This script is based on https://huggingface.co/datasets/tokyotech-llm/lmsys-chat-1m-synth/blob/main/materials/generate_assistant_responses.py.
"""

import argparse
import os, copy
import json
import gzip
import concurrent.futures
import time
from tqdm import tqdm
from openai import OpenAI

def process_record(openai_instance, record, model, n, temperature, top_p, max_tokens, max_retries, frequency_penalty, system_message, target_language, no_thinking):
    """
    Generates n responses for a given record using DeepInfra API, with optional system message.
    """
    if target_language == "Japanese":
        messages = record["translated_conversation"]
    elif target_language == "English":
        messages = [{"role": record["conversation"][0]["role"], 
                    "content": record["conversation"][0]["content"]}]
    else:
        raise ValueError(f"Unsupported language: {target_language}")
    
    REPETITION_THRESHOLD = 2000
    
    # specify the system_message
    if system_message:
        messages = [{"role": "system", "content": system_message}] + messages

    synthesized_responses = []
    synthesized_response_scoring_annotations = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for _ in range(n):
        retry_count = 0
        while retry_count < max_retries:
            try:
                if no_thinking:
                    chat_completion = openai_instance.chat.completions.create(
                            model=model,
                            max_tokens=max_tokens,
                            messages=messages,
                            temperature=temperature,
                            top_p=top_p,
                            frequency_penalty=frequency_penalty,
                            n=1,
                            extra_body={
                                "chat_template_kwargs": {"enable_thinking": False},
                            },
                            )
                    response = chat_completion.choices[0].message.content.strip()
                    synthesized_responses.append({"role": "assistant", "content": response})


                else:
                    chat_completion = openai_instance.chat.completions.create(
                            model=model,
                            max_tokens=max_tokens,
                            messages=messages,
                            temperature=temperature,
                            top_p=top_p,
                            frequency_penalty=frequency_penalty,
                            n=1,
                            )
                
                    # response
                    response = chat_completion.choices[0].message.content
                    try:
                        reasoning_content = chat_completion.choices[0].message.reasoning_content.strip()
                        synthesized_responses.append({"role": "assistant", "content": response, "reasoning_content": reasoning_content})
                    except:
                        synthesized_responses.append({"role": "assistant", "content": response})
                
                # token count, repetition flag
                is_repetition_in_response = chat_completion.usage.completion_tokens > REPETITION_THRESHOLD
                dict_response_scoring = {
                    "response_token_length": chat_completion.usage.completion_tokens,
                    "repetition_flag": is_repetition_in_response,
                }
                synthesized_response_scoring_annotations.append(dict_response_scoring)
                
                total_prompt_tokens += chat_completion.usage.prompt_tokens
                total_completion_tokens += chat_completion.usage.completion_tokens
                break
            except Exception as e:
                print(e)
                print(f"Retry {retry_count + 1}/{max_retries} failed.")
                retry_count += 1
                time.sleep(5)
        
        if retry_count == max_retries:
            print(f"Max retries reached for a response. Skipping...")
            continue

    record["synthesized_assistant_responses"] = synthesized_responses
    record["synthesized_response_scoring_annotations"] = synthesized_response_scoring_annotations
    record["synthesis_model"] = model

    return record, total_prompt_tokens, total_completion_tokens

def main(input_path, output_path, api_key, model, start_index, end_index, test_run, append, n, max_retries, temperature, top_p,
         max_tokens, num_threads, api_base_url, frequency_penalty, system_message, target_language, no_thinking):
    openai_instance = OpenAI(api_key=api_key, base_url=api_base_url)

    if append:
        ofs_mode = "at"
    else:
        ofs_mode = "wt"

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_responses = 0

    with gzip.open(input_path, mode="rb") as ifs, gzip.open(output_path, mode=ofs_mode) as ofs:
        idx = 0
        total_records = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []

            for line in tqdm(ifs, desc="Processing records"):
                print(idx)
                if idx < start_index:
                    idx += 1
                    continue
                
                if idx >= end_index:
                    break

                record = json.loads(line.strip())

                if record.get("empty_user_instruction_ja", False) or record.get("duplicate_user_instruction_ja") is not None or record.get("redacted", False):
                    idx += 1
                    processed_record = copy.deepcopy(record)
                    processed_record["synthesized_assistant_responses"] = None
                    processed_record["synthesized_response_scoring_annotations"] = None
                    record["synthesis_model"] = None
                    ofs.write(json.dumps(processed_record, ensure_ascii=False) + "\n")
                    continue

                # submit tasks
                future = executor.submit(process_record, openai_instance, record, model, n, temperature, top_p, max_tokens, max_retries, frequency_penalty, system_message, target_language, no_thinking)
                futures.append(future)

                if len(futures) >= num_threads:
                    done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

                    for completed_future in done:
                        processed_record, prompt_tokens, completion_tokens = completed_future.result()
                        ofs.write(json.dumps(processed_record, ensure_ascii=False) + "\n")
                        ofs.flush()
                        if test_run:
                            total_prompt_tokens += prompt_tokens
                            total_completion_tokens += completion_tokens
                            total_responses += n
                            
                    futures = list(not_done)

                idx += 1
                total_records += 1

                if test_run and total_records >= 500:
                    print("Test run complete.")
                    break

            for completed_future in concurrent.futures.as_completed(futures):
                processed_record, prompt_tokens, completion_tokens = completed_future.result()
                ofs.write(json.dumps(processed_record, ensure_ascii=False) + "\n")
                ofs.flush()
                if test_run:
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    total_responses += n

    if test_run and total_responses > 0:
        print(f"Test run statistics:")
        print(f"Total prompt tokens: {total_prompt_tokens}")
        print(f"Total completion tokens: {total_completion_tokens}")
        print(f"Average prompt tokens per response: {total_prompt_tokens / total_responses}")
        print(f"Average completion tokens per response: {total_completion_tokens / total_responses}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch generate responses using vLLM API.")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the input JSONL.gz file.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to the output JSONL.gz file.")
    parser.add_argument('--api_key', type=str, required=True, help="API key.")
    parser.add_argument('--api_base_url', type=str, required=True, help="API base url.")
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model to use for generation.")
    parser.add_argument('--start_index', type=int, default=0, help="Index to start processing from.")
    parser.add_argument('--end_index', type=int, default=200000, help="Index to end processing at.")
    parser.add_argument('--test_run', action='store_true', help="Run in test mode (process only first 300 conversations).")
    parser.add_argument('--append', action='store_true', help="Append to the output file instead of overwriting.")
    parser.add_argument('--n', type=int, required=True, help="Number of responses to generate per user instruction.")
    parser.add_argument('--max_retries', type=int, default=3, help="Maximum number of retries for failed requests.")
    parser.add_argument('--temperature', type=float, default=0.6, help="Sampling temperature for response generation.")
    parser.add_argument('--top_p', type=float, default=1.0, help="Nucleus sampling probability for response generation.")
    parser.add_argument('--max_tokens', type=int, default=4096, help="Number of maximum generated tokens.")
    parser.add_argument('--num_threads', type=int, default=200, help="Number of threads for parallel processing.")
    parser.add_argument('--frequency_penalty', type=float, default=0.0, help="Penalty for repeated tokens.")
    parser.add_argument('--system_message', type=str, default=None, help="System message to prepend to the conversation.")
    parser.add_argument('--target_language', type=str, default="Japanese", choices=["Japanese", "English"], 
                        help="Target conversation language. It affects which user instruction will be used.")
    parser.add_argument('--no_thinking', action='store_true', help="thinking or not")

    args = parser.parse_args()
    main(input_path=args.input_path, output_path=args.output_path, api_key=args.api_key, model=args.model, 
        start_index=args.start_index, end_index=args.end_index, test_run=args.test_run, append=args.append, n=args.n, 
        max_retries=args.max_retries, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, 
        num_threads=args.num_threads, api_base_url=args.api_base_url, frequency_penalty=args.frequency_penalty, 
        system_message=args.system_message, target_language=args.target_language,
         no_thinking=args.no_thinking)

