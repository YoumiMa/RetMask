import torch
import os
import numpy as np
import argparse
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from collections import Counter
import json
import gc

def get_mask_heads(model_name: str, 
                   num_layers: int,
                   num_heads: int,
                   topk: int, 
                   mask_greater_than: float = 0, 
                   score_path: str = None, 
                   random_mask: bool = False):
    """Load and return heads to mask"""
    if topk == 0 and  mask_greater_than == 0:
        return None
    
    if score_path: # read from a score file other than the default one stored in head_score/{model_name}.json
        with open(score_path) as file:
            stable_block_list =  json.loads(file.readline())  
            stable_block_list = [(l[0], np.mean(l[1])) for l in stable_block_list.items()] 
            stable_block_list = sorted(stable_block_list, key=lambda x: x[1], reverse=True) 
            retrieval_head_list = [[int(ll) for ll in l[0].split("-")] for l in stable_block_list]
    else:
        with open(f"head_score/{model_name}.json", "r") as file:
            stable_block_list =  json.loads(file.readline())  
            stable_block_list = [(l[0], np.mean(l[1])) for l in stable_block_list.items()] 
            stable_block_list = sorted(stable_block_list, key=lambda x: x[1], reverse=True) 
            retrieval_head_list = [[int(ll) for ll in l[0].split("-")] for l in stable_block_list]  

    if mask_greater_than > 0:
        topk = len([l for l in stable_block_list if l[1] >= mask_greater_than])
        # topk = int(len(stable_block_list) * mask_greater_than) # top mask_greater_than%
    
    elif mask_greater_than < 0:
        topk = - len([l for l in stable_block_list if l[1] >= - mask_greater_than])
        # topk = int(len(stable_block_list) * mask_greater_than) # top mask_greater_than%
        
    if topk > 0:
        print(f"Masking out top {topk} retrieval heads")
        return retrieval_head_list[:topk]
    else:
        print(f"Masking out random {-topk} heads")
        # Construct random heads
        results = []
        while len(results) < -topk:
            l = random.randrange(num_layers)
            h = random.randrange(num_heads)
            if not random_mask: # randomly mask non-retrieval heads
                if (l, h) not in results and [l, h] not in retrieval_head_list[:-topk]:
                    results.append((l, h))
            else: # ranndomly mask any head
                if (l, h) not in results:
                    results.append((l, h))
        return results


def mask_model_heads(save_dir: str, 
                     model_path: str, 
                     masked_heads: list,
                     is_random=False, 
                     score_path: str = None, 
                     random_mask: bool = False):
    """Mask attention heads in the model and save it"""
    if not masked_heads:
        return model_path
    
    if is_random:
        if random_mask:
            output_path = os.path.join(save_dir, f"{model_path.replace('/', '_')}_masked_{len(masked_heads)}heads_random_all")
        else:
            output_path = os.path.join(save_dir, f"{model_path.replace('/', '_')}_masked_{len(masked_heads)}heads_random")
    else:
        output_path = os.path.join(save_dir, f"{model_path.replace('/', '_')}_masked_{len(masked_heads)}heads")
    
    if score_path:
        # output_path += "_QRhead"
        output_path += "_aligned"
        
    if os.path.exists(output_path):
        print(f"✓ Using cached masked model: {output_path}")
        return output_path
    
    print(f"Loading model to mask heads...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        raise AttributeError("Cannot find model layers")
    
    print(f"Masking {len(masked_heads)} heads...")
    masked_count = 0
    print(masked_heads)
    for layer_idx, head_idx in tqdm(masked_heads, desc="Masking heads"):
        if layer_idx >= len(layers):
            continue
        
        layer = layers[layer_idx]
        attn = layer.self_attn if hasattr(layer, 'self_attn') else None
        
        if attn is None:
            continue
        
        # Get head_dim
        if hasattr(attn, 'head_dim'):
            head_dim = attn.head_dim
        else:              
            num_heads = model.config.num_attention_heads
            hidden_size = model.config.hidden_size
            head_dim = hidden_size // num_heads
        
        start = head_idx * head_dim
        end = (head_idx + 1) * head_dim
        
        with torch.no_grad():
            if hasattr(attn, 'o_proj'):
                attn.o_proj.weight.data[:, start:end] = 0
                if attn.o_proj.bias is not None:
                    attn.o_proj.bias.data[start:end] = 0
                masked_count += 1
    
    print(f"Successfully masked {masked_count}/{len(masked_heads)} heads")
    
    # Save masked model
    print(f"Saving masked model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"✓ Masked model saved")
    
    # Clean up
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--save_dir', type=str, default='./')
    parser.add_argument('--mask_topk', type=int, default=0)
    parser.add_argument('--mask_greater_than', type=float, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument('--random', action="store_true", help="Mask heads totally randomly")
    parser.add_argument("--head_score_path", type=str, default=None, help="Path to head scores if not using those in head_score.")
    args = parser.parse_args()
    
    random_seed = 42
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    
    model_version = args.model_name.split("/")[-1]

    config = AutoConfig.from_pretrained(args.model_name)
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    
    # Get heads to mask
    masked_heads = get_mask_heads(model_version, 
                                  num_layers=num_layers,
                                  num_heads=num_heads,
                                  topk=args.mask_topk, 
                                  mask_greater_than=args.mask_greater_than, 
                                  score_path=args.head_score_path,
                                  random_mask=args.random)
    
    # Prepare model (mask if needed)
    if masked_heads:
        model_path = mask_model_heads(save_dir=args.save_dir, 
                                      model_path=args.model_name, 
                                      masked_heads=masked_heads, 
                                      is_random=args.mask_topk < 0 or args.mask_greater_than < 0, 
                                      score_path=args.head_score_path, 
                                      random_mask=args.random)
        if args.mask_topk > 0:
            save_name = f"{model_version}_block_top{len(masked_heads)}"
        else:
            save_name = f"{model_version}_block_random{len(masked_heads)}"
    else:
        model_path = args.model_name
        save_name = model_version



if __name__ == '__main__':
    main()
