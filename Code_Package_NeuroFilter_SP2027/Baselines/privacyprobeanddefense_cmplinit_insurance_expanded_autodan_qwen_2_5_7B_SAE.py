#!/usr/bin/env python

# --------------------------------------------------------------------------
# Section 0: Imports & Configuration
# --------------------------------------------------------------------------
import os
import torch
import torch.nn as nn
import torch.optim as optim
import gc
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

# --- GLOBAL SETTINGS ---
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct" 
CHECKPOINT_BASE = "/path/to/code/checkpoints/qwen7b_saes_tinystories_torch"
FIGURE_PATH = '/path/to/code/figures'
DATAFRAME_PATH = '/path/to/code/dataframes'

# SAE Training Params
TOKENS_PER_LAYER = 1_000_000 
BATCH_SIZE = 512 
SAE_EXPANSION_FACTOR = 4
SAE_L1_COEFF = 5e-4
LR = 3e-4
MAX_ACTIVATIONS_IN_RAM = 4096 

os.makedirs(CHECKPOINT_BASE, exist_ok=True)
os.makedirs(FIGURE_PATH, exist_ok=True)
os.makedirs(DATAFRAME_PATH, exist_ok=True)

# --------------------------------------------------------------------------
# Section 0.5: SAE Class
# --------------------------------------------------------------------------
class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, expansion_factor=4):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_model * expansion_factor
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_model, self.d_sae)))
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.d_sae, d_model)))
        self.b_dec = nn.Parameter(torch.zeros(d_model)) 
        with torch.no_grad():
            self.W_dec.data /= self.W_dec.data.norm(dim=0, keepdim=True)

    def forward(self, x):
        x_cent = x - self.b_dec
        acts = torch.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        return x_reconstruct, acts

# --------------------------------------------------------------------------
# Section 1: Memory-Hardened Pipeline
# --------------------------------------------------------------------------
def get_activations_hook(buffer_list):
    def hook(module, input, output):
        # Flatten and keep on CPU in float16 to save RAM
        acts = output[0].detach().view(-1, output[0].shape[-1]).to(torch.float16).cpu()
        buffer_list.append(acts)
    return hook

def run_layer_pipeline(model, tokenizer, layer_idx, tr_data, te_data):
    d_model = model.config.hidden_size
    save_path = os.path.join(CHECKPOINT_BASE, f"layer_{layer_idx}.pt")
    
    # -------------------------------------------------------
    # 1. Training Logic (SKIP IF EXISTS)
    # -------------------------------------------------------
    print(f'Checking for checkpoint at {save_path}')
    if os.path.exists(save_path):
        print(f"\n⏩ Checkpoint found for Layer {layer_idx}. Skipping training.")
    else:
        print(f"\n🚀 Training SAE: Layer {layer_idx}")
        sae = SparseAutoencoder(d_model, SAE_EXPANSION_FACTOR).to("cuda")
        optimizer = optim.Adam(sae.parameters(), lr=LR)
        
        # Load TinyStories Streaming
        dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        data_iter = iter(dataset)
        
        activation_buffer = []
        hook_handle = model.model.layers[layer_idx].register_forward_hook(get_activations_hook(activation_buffer))
        
        tokens_processed = 0
        pbar = tqdm(total=TOKENS_PER_LAYER, desc=f"L{layer_idx} Train")
        
        while tokens_processed < TOKENS_PER_LAYER:
            # Get just enough activations for one training cycle
            while sum(len(x) for x in activation_buffer) < MAX_ACTIVATIONS_IN_RAM:
                try:
                    batch_text = [next(data_iter)['text'] for _ in range(4)]
                    inputs = tokenizer(batch_text, return_tensors="pt", truncation=True, max_length=128, padding=True).to(model.device)
                    with torch.no_grad(): model(**inputs)
                except StopIteration:
                    data_iter = iter(dataset)
            
            flat_acts = torch.cat(activation_buffer, dim=0)
            activation_buffer.clear() # Immediate clear
            
            # Sub-batch loop to keep VRAM spikes low
            for i in range(0, len(flat_acts), BATCH_SIZE):
                if tokens_processed >= TOKENS_PER_LAYER: break
                batch = flat_acts[i:i+BATCH_SIZE].to("cuda", non_blocking=True).to(torch.float32)
                
                optimizer.zero_grad()
                recon, acts = sae(batch)
                loss = (batch - recon).pow(2).sum(-1).mean() + SAE_L1_COEFF * acts.sum(-1).mean()
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    sae.W_dec.data /= sae.W_dec.data.norm(dim=0, keepdim=True)
                
                tokens_processed += batch.shape[0]
                pbar.update(batch.shape[0])
            
            del flat_acts # Aggressive cleaning
            
        hook_handle.remove()
        torch.save(sae.state_dict(), save_path)
        pbar.close()
        del sae, optimizer, dataset, data_iter
        gc.collect()
        torch.cuda.empty_cache()

    # -------------------------------------------------------
    # 2. Probing (Always Load from Disk)
    # -------------------------------------------------------
    print(f"🔬 Probing Layer {layer_idx}...")
    sae = SparseAutoencoder(d_model, SAE_EXPANSION_FACTOR).to("cuda")
    # This load is safe because we either just saved it (training) or found it (skipped)
    sae.load_state_dict(torch.load(save_path))
    sae.eval()

    def extract_and_process(data_tuple):
        prompts, meta = data_tuple
        acts_list = []
        for chat in prompts:
            tokens = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(tokens, output_hidden_states=True)
                # Keep only what we need in float16 on CPU
                acts_list.append(out.hidden_states[layer_idx+1][:, -1, :].to(torch.float16).cpu())
        return torch.cat(acts_list)

    X_tr_acts = extract_and_process(tr_data).cuda().float()
    y_tr = torch.tensor([1 if m['type'] == 'privacy_violation' else 0 for m in tr_data[1]]).cuda()
    
    with torch.no_grad():
        _, c_tr = sae(X_tr_acts)
        probe = (c_tr[y_tr==1].mean(0) - c_tr[y_tr==0].mean(0)) @ sae.W_dec
    
    X_te_acts = extract_and_process(te_data).cuda().float()
    y_te = np.array([1 if m['type'] == 'privacy_violation' else 0 for m in te_data[1]])
    
    scores_tr = (X_tr_acts @ probe).cpu().numpy()
    scores_te = (X_te_acts @ probe).cpu().numpy()
    thr = (scores_tr[y_tr.cpu()==1].mean() + scores_tr[y_tr.cpu()==0].mean()) / 2.0
    acc = accuracy_score(y_te, (scores_te > thr).astype(int))
    
    layer_scores = [{"Layer": layer_idx, "Score": s, "Type": te_data[1][i]['type']} for i, s in enumerate(scores_te)]
    
    # Cleanup before next layer
    del sae, X_tr_acts, X_te_acts, probe, c_tr
    gc.collect()
    torch.cuda.empty_cache()
    
    return acc, layer_scores

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Max quantization for stability
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # Load Insurance (Scenario 2)
    PROFILES_DIR = "/path/to/code/benchmark/datasets"
    with open(f'{PROFILES_DIR}/generated_profiles_scenario_2.json', 'r') as f:
        stories = [json.dumps(p, indent=2) for p in json.load(f)]
    with open("/path/to/code/logs/autodan_style_matched_benchmark_insurance_10_per_type_nonuniformscores.json", 'r') as f:
        b_data = json.load(f)
    
    tr_s, te_s = train_test_split(stories, test_size=0.3, random_state=42)
    def prep(sty, priv, ben):
        p, m = [], []
        for s in sty:
            sys = f"You are Agent Safety... {s}"
            for txt in priv: p.append([{"role":"system","content":sys},{"role":"user","content":txt}]); m.append({"type":"privacy_violation"})
            for txt in ben: p.append([{"role":"system","content":sys},{"role":"user","content":txt}]); m.append({"type":"benign"})
        return p, m

    tr_data = prep(tr_s, b_data['privacy_violating_prompts'], b_data['benign_prompts'])
    te_data = prep(te_s, b_data['privacy_violating_prompts'], b_data['benign_prompts'])

    # Processing later layers (32 to 63)
    LAYERS_TO_PROCESS = list(range(14, 28)) 
    results_acc, results_scores = [], []

    for l_idx in LAYERS_TO_PROCESS:
        try:
            acc, scores = run_layer_pipeline(model, tokenizer, l_idx, tr_data, te_data)
            results_acc.append(acc)
            results_scores.extend(scores)
        except Exception as e:
            print(f"⚠️ Error on Layer {l_idx}: {e}")
            continue

    # Plotting
    df = pd.DataFrame(results_scores)
    df.to_csv(f"{DATAFRAME_PATH}/qwen7B_sae_final_nonuniform_may22.csv", index=False)
    plt.rcParams.update({'font.size': 18})
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(results_acc)), results_acc, marker='o')
    plt.xlabel("Layer Index (14-27)")
    plt.ylabel("Accuracy")
    plt.savefig(f"{FIGURE_PATH}/qwen7B_sae_acc_CMPL_Insurance_nonuniform_may22.pdf", format='pdf', bbox_inches='tight')
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="Layer", y="Score", hue="Type")
    plt.savefig(f"{FIGURE_PATH}/qwen7B_sae_projection_scores_CMPL_Insurance_nonuniform_may22.pdf", format='pdf', bbox_inches='tight')
    
    print("✅ Pipeline Complete!")