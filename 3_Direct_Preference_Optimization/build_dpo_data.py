import json, gzip
import argparse
from tqdm import tqdm

def load_jsonl(path: str):
    data = []
    with gzip.open(path, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def dump_jsonl(data: list, path: str):
    with gzip.open(path, "wt") as wf:
        for d in data:
            wf.write(json.dumps(d, ensure_ascii=False) + "\n")
    return

def main(chosen_path: str, rejected_path: str, output_path: str):

    print(f"Loading data file with responses to choose...")
    data_chosen = load_jsonl(chosen_path)

    print(f"Loading data file with responses to rejected...")
    data_rejected = load_jsonl(rejected_path)

    cid2data_rejected = {d['conversation_id']: d for d in data_rejected}
    
    dpo_data = []
    
    for d in tqdm(data_chosen, desc="Building DPO data"):
        cid = d["conversation_id"]
        d_rejected = cid2data_rejected[cid]
        curr_d = {"conversation_id": cid}
        instruction = {"role": d_rejected["conversation"][0]["role"], "content": d_rejected["conversation"][0]["content"]}
        
        if d["synthesized_assistant_responses"] and  d_rejected["synthesized_assistant_responses"]:
            if d["redacted"]:
                continue
            # if there are multiple responses generated, use the first non-null response (basically random sampling)
            for i in range(len(d["synthesized_assistant_responses"])):
                resp_chosen = d["synthesized_assistant_responses"][i]
                if resp_chosen["content"] != None:
                    break
            for i in range(len(d_rejected["synthesized_assistant_responses"])):
                resp_rejected = d_rejected["synthesized_assistant_responses"][i]
                if resp_rejected["content"] != None:
                    break
            if resp_chosen["content"] == None or resp_rejected["content"] == None:
                continue
            if resp_chosen["content"] == resp_rejected["content"]:
                continue
            curr_d["chosen"] = [instruction, resp_chosen]
            curr_d["rejected"] = [instruction, resp_rejected]
        
            dpo_data.append(curr_d)

    print(f"DPO data built, saving to {output_path}...")
    
    dump_jsonl(dpo_data, output_path)
    
    return
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch generate responses using DeepInfra API.")
    parser.add_argument('--chosen_path', type=str, required=True, help="Path to the input JSONL.gz file for chosen responses.")
    parser.add_argument('--rejected_path', type=str, required=True, help="Path to the input JSONL.gz file for rejected responses.")
    parser.add_argument('--output_path', type=str, required=True, help="Output path to store data for DPO training.")
    args = parser.parse_args()

    main(chosen_path=args.chosen_path, rejected_path=args.rejected_path, output_path=args.output_path)