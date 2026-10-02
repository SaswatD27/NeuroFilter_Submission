# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import numpy as np
import gc
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import hashlib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score  # <-- Added import for accuracy

# --- 1. Model & Environment Setup ---
model_id = "meta-llama/Llama-3.3-70B-Instruct"  # "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
print(f"--- Loading Model: {model_id} ---")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("\n✅ Model loaded successfully.")

# Define file paths and directories
ATTACK_FILE = '/path/to/code/notebooks/FRACTURED-SORRY-Bench-Automated-Multishot-Jailbreaking/data/question_breakdowns.jsonl'
BENIGN_FILE = '/path/to/code/notebooks/FRACTURED-SORRY-Bench-Automated-Multishot-Jailbreaking/data/benign_questions_decomposed.jsonl'

# --- Define Directories for outputs ---
FIGURE_DIR = '/path/to/code/figures'
DF_DIR = '/path/to/code/dataframes' # Define directory for CSVs
model_id_alt = model_id.split('/')[-1].replace('.', '_').replace('-', '_') # Create a filesystem-safe model ID
RUN_SUFFIX = "fractured_sorry_full_layerthreshold"
# Use a dedicated cache dir for the k-fold run (this script uses different cache keys).
TEMP_DIR = f"./temp_stateful_deltas_{model_id_alt}_{RUN_SUFFIX}_kfold" # Define temp directory for deltas

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(DF_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True) # Create TEMP_DIR, don't remove if it exists
print(f"✅ Output directories set:")
print(f"   Figures: {FIGURE_DIR}")
print(f"   DataFrames: {DF_DIR}")
print(f"   Temp Deltas: {TEMP_DIR}")

# --- 1.5. Cross-validation & threshold config ---
N_SPLITS = 5
RANDOM_SEED = 42
FINAL_TEST_SIZE = 0.40  # match original script's 60/40 train-test split
THRESHOLD_GRID_POINTS = 101
THRESHOLD_RANGE_LO_Q = 0.01
THRESHOLD_RANGE_HI_Q = 0.99
THRESHOLD_RANGE_PAD_FRAC = 0.10
# Metric used for threshold selection:
# - "macro_turn": mean accuracy across turns (each turn weighted equally)
# - "micro": accuracy across all (turn,row) samples (longer convos weighted more)
THRESHOLD_SELECT_METRIC = "macro_turn"


# --- 2. Data Loading and K-Fold Setup ---
def load_mosaic_conversations(filepath): #, max_items=100): # Reduced size for faster iteration
    conversations = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                # if len(conversations) >= max_items: breaks
                data = json.loads(line)
                # Ensure we only take conversations with 2+ steps (to have at least one delta)
                if data.get('decomposed_steps') and len(data['decomposed_steps']) > 1:
                    conversations.append(data['decomposed_steps'])
        print(f"✅ Loaded {len(conversations)} conversations from {os.path.basename(filepath)}")
    except Exception as e:
        print(f"⚠️ Error loading {filepath}: {e}")
    return conversations

# Load both datasets first
all_attack_convos = load_mosaic_conversations(ATTACK_FILE) #, max_items=100) # Respect the user's max_items
all_benign_convos = load_mosaic_conversations(BENIGN_FILE) #, max_items=100)

# Find the minimum number to ensure balanced data
num_convos = min(len(all_attack_convos), len(all_benign_convos))

if num_convos == 0:
    print("⚠️ Critical Error: At least one conversation type (attack or benign) failed to load any data. Halting.")
    # Set probe_trained to False to skip downstream steps gracefully
    probe_trained = False
    balanced_attack_convos, balanced_benign_convos = [], []
elif num_convos < 10: # Just a warning
    print(f"⚠️ Warning: Very few conversations loaded ({num_convos}). Results may be unstable.")
else:
    print(f"--- Balancing data: Using {num_convos} conversations of each type. ---")

# Only proceed if we have data
if num_convos > 0:
    # Truncate lists to the minimum
    balanced_attack_convos = all_attack_convos[:num_convos]
    balanced_benign_convos = all_benign_convos[:num_convos]
else:
    balanced_attack_convos, balanced_benign_convos = [], []

print("\n--- 2.5 Creating fixed 60/40 train-test split (conversation-level) ---")
if num_convos <= 0:
    attack_train_convos, benign_train_convos = [], []
    attack_test_convos, benign_test_convos = [], []
else:
    all_convos = balanced_attack_convos + balanced_benign_convos
    y_convos_all = np.concatenate([np.ones(len(balanced_attack_convos)), np.zeros(len(balanced_benign_convos))]).astype(int)
    idx_all = np.arange(len(all_convos))

    train_idx, test_idx = train_test_split(
        idx_all, test_size=FINAL_TEST_SIZE, random_state=RANDOM_SEED, stratify=y_convos_all
    )
    train_idx = train_idx.tolist()
    test_idx = test_idx.tolist()

    attack_train_convos = [all_convos[i] for i in train_idx if y_convos_all[i] == 1]
    benign_train_convos = [all_convos[i] for i in train_idx if y_convos_all[i] == 0]
    attack_test_convos = [all_convos[i] for i in test_idx if y_convos_all[i] == 1]
    benign_test_convos = [all_convos[i] for i in test_idx if y_convos_all[i] == 0]

    print(f"📊 Fixed split: {len(attack_train_convos)} attack & {len(benign_train_convos)} benign for training.")
    print(f"📊 Fixed split: {len(attack_test_convos)} attack & {len(benign_test_convos)} benign for final testing.")


# --- 3. Activation Extraction & Probe Training (with Full State & Caching) ---

def stable_convo_key(shots, convo_type):
    """
    Stable cache key for a conversation, independent of fold/split enumeration.
    Includes convo_type to avoid cross-class collisions.
    """
    payload = (convo_type + "\n" + "\n".join(shots)).encode("utf-8", errors="ignore")
    return hashlib.sha1(payload).hexdigest()[:16]

def get_activations_and_response(model, tokenizer, conversation_history, new_shot, device='cuda'):
    """
    Gets activations for a prompt AND generates the model's response to build a truly stateful history.
    """
    prompt_chat = conversation_history + [{"role": "user", "content": new_shot}]
    try:
        encoded = tokenizer.apply_chat_template(
            prompt_chat, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, return_tensors="pt")
    if isinstance(encoded, torch.Tensor):
        input_ids = encoded.to(device)
        model_inputs = {"input_ids": input_ids}
    else:
        encoded = encoded.to(device)
        if "input_ids" not in encoded:
            raise ValueError("Tokenizer encoding missing 'input_ids'.")
        input_ids = encoded["input_ids"]
        model_inputs = dict(encoded)
    with torch.no_grad():
        outputs = model(**model_inputs, output_hidden_states=True)
    # Get activations from all hidden states for the last token
    activations = torch.stack([h[0, -1, :].squeeze().cpu().float() for h in outputs.hidden_states]).numpy()
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    prompt_len = int(input_ids.shape[-1])
    response_text = tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True).strip()
    del outputs, encoded, generated_ids
    gc.collect()
    torch.cuda.empty_cache()
    return activations, response_text

def get_or_generate_deltas(conversations, split_name, convo_type, temp_dir, desc):
    """
    Processes conversations to get deltas, loading from temp_dir if possible.
    Saves deltas per conversation.
    Returns a single stacked numpy array of all deltas.
    """
    all_deltas_list = [] # This will be a list of np arrays, one per convo
    
    # Handle empty input
    if not conversations:
        print(f"--- No {split_name} {convo_type} conversations to process. ---")
        return np.array([])
    
    print(f"--- Processing {split_name} {convo_type} deltas ---")
    for convo_id, shots in enumerate(tqdm(conversations, desc=desc)):
        convo_key = stable_convo_key(shots, convo_type)
        filepath = os.path.join(temp_dir, f"{convo_type}_{convo_key}.npy")
        
        # --- 1. Try to load from cache ---
        if os.path.exists(filepath):
            try:
                convo_deltas_np = np.load(filepath)
                all_deltas_list.append(convo_deltas_np)
                continue # Successfully loaded, skip to next conversation
            except Exception as e:
                print(f"⚠️ Warning: Could not load {filepath}, regenerating. Error: {e}")

        # --- 2. If loading fails, generate deltas ---
        convo_deltas_list_turns = [] # Deltas for this one conversation
        conversation_history = []
        
        # First turn
        try:
            prev_activations, response = get_activations_and_response(model, tokenizer, [], shots[0])
            conversation_history.extend([{"role": "user", "content": shots[0]}, {"role": "assistant", "content": response}])
        except Exception as e:
            print(f"Error on first turn for {filepath}: {e}. Skipping convo.")
            continue

        # Subsequent turns
        for shot in shots[1:]:
            try:
                current_activations, response = get_activations_and_response(model, tokenizer, conversation_history, shot)
                delta = current_activations - prev_activations
                convo_deltas_list_turns.append(delta) # Append the delta (a numpy array)
                prev_activations = current_activations
                conversation_history.extend([{"role": "user", "content": shot}, {"role": "assistant", "content": response}])
            except Exception as e:
                print(f"Error on subsequent turn for {filepath}: {e}. Skipping turn.")
                break # Stop processing this conversation
        
        # --- 3. Save to cache ---
        if convo_deltas_list_turns:
            # Stack all turns for this conversation into a single numpy array
            convo_deltas_np = np.stack(convo_deltas_list_turns, axis=0) # Shape: (Turns, Layers, Dim)
            np.save(filepath, convo_deltas_np)
            all_deltas_list.append(convo_deltas_np)
        else:
            print(f"No deltas generated for {filepath}")

    # --- 4. Concatenate all conversations into one dataset ---
    if not all_deltas_list:
        print(f"⚠️ No deltas found or generated for {split_name} {convo_type}.")
        return np.array([]) # Return empty array
        
    return np.concatenate(all_deltas_list, axis=0) # Concatenate along the 'turns' dimension

def ensure_deltas_cached(conversations, split_name, convo_type, temp_dir, desc):
    """
    Ensures per-conversation delta files exist on disk.

    Unlike `get_or_generate_deltas`, this does NOT load/concatenate cached deltas
    into RAM; it only runs the model for conversations missing cache files (or
    with unreadable cache files).
    """
    if not conversations:
        print(f"--- No {split_name} {convo_type} conversations to cache. ---")
        return

    cache_hits = 0
    cache_generated = 0
    cache_regenerated = 0

    print(f"--- Ensuring cache for {split_name} {convo_type} deltas ---")
    for shots in tqdm(conversations, desc=desc):
        convo_key = stable_convo_key(shots, convo_type)
        filepath = os.path.join(temp_dir, f"{convo_type}_{convo_key}.npy")

        if os.path.exists(filepath):
            try:
                _ = np.load(filepath, mmap_mode="r")
                cache_hits += 1
                continue
            except Exception:
                cache_regenerated += 1

        # Missing or unreadable cache -> generate deltas.
        convo_deltas_list_turns = []
        conversation_history = []

        try:
            prev_activations, response = get_activations_and_response(model, tokenizer, [], shots[0])
            conversation_history.extend(
                [{"role": "user", "content": shots[0]}, {"role": "assistant", "content": response}]
            )
        except Exception as e:
            print(f"Error on first turn for {filepath}: {e}. Skipping convo.")
            continue

        for shot in shots[1:]:
            try:
                current_activations, response = get_activations_and_response(model, tokenizer, conversation_history, shot)
                delta = current_activations - prev_activations
                convo_deltas_list_turns.append(delta)
                prev_activations = current_activations
                conversation_history.extend(
                    [{"role": "user", "content": shot}, {"role": "assistant", "content": response}]
                )
            except Exception as e:
                print(f"Error on subsequent turn for {filepath}: {e}. Skipping turn.")
                break

        if convo_deltas_list_turns:
            convo_deltas_np = np.stack(convo_deltas_list_turns, axis=0)
            np.save(filepath, convo_deltas_np)
            cache_generated += 1
        else:
            print(f"No deltas generated for {filepath}")

    print(
        f"✅ Cache summary for {split_name} {convo_type}: "
        f"{cache_hits} hit(s), {cache_generated} generated, {cache_regenerated} regenerated."
    )

def train_differential_probe(X_probe_train, y_probe_train, random_seed=RANDOM_SEED):
    """
    Trains a per-layer logistic regression probe and returns the best layer + weights.
    Uses training accuracy to pick the best layer (consistent with the original script).
    """
    num_layers = X_probe_train.shape[1]
    probe_accuracies = []
    probe_weights = []
    print(f"\n--- Training a 'Truly Stateful Differential Probe' for {num_layers} layers ---")
    for layer in tqdm(range(num_layers), desc="Training Probes"):
        probe = LogisticRegression(max_iter=1000, random_state=random_seed, class_weight='balanced')
        probe.fit(X_probe_train[:, layer, :], y_probe_train)
        probe_accuracies.append(probe.score(X_probe_train[:, layer, :], y_probe_train))
        probe_weights.append(probe.coef_[0])

    max_accuracy = float(np.max(probe_accuracies))
    candidate_indices = np.where(np.array(probe_accuracies) == max_accuracy)[0]
    best_layer_index = int(candidate_indices[-1])
    best_probe_weights = probe_weights[best_layer_index]
    return best_layer_index, best_probe_weights, probe_accuracies


def train_all_layer_differential_probes(X_probe_train, y_probe_train, random_seed=RANDOM_SEED):
    """
    Trains one logistic-regression probe per layer and returns all weights.
    Layer selection is left to validation, not train accuracy.
    """
    num_layers = X_probe_train.shape[1]
    probe_accuracies = []
    probe_weights = []
    print(f"\n--- Training layer-candidate probes for {num_layers} layers ---")
    for layer in tqdm(range(num_layers), desc="Training Layer Candidates"):
        probe = LogisticRegression(max_iter=1000, random_state=random_seed, class_weight='balanced')
        probe.fit(X_probe_train[:, layer, :], y_probe_train)
        probe_accuracies.append(float(probe.score(X_probe_train[:, layer, :], y_probe_train)))
        probe_weights.append(probe.coef_[0])
    return probe_weights, probe_accuracies


# --- 4. Stateful Cumulative Scoring Helper (from CACHE) ---
print("\n--- 4. Defining Cumulative Scoring Helper (from Cache) ---")

def analyze_cumulative_from_cache(conversations, split_name, convo_type, temp_dir, best_probe_weights, best_layer_index):
    """
    Analyzes test conversations by LOADING cached deltas and tracking the cumulative projection score.
    """
    results = []
    
    # Handle empty input
    if not conversations:
        print(f"--- No {split_name} {convo_type} conversations to analyze. ---")
        return []
    
    # Ensure deltas are cached first (this call will be fast if files exist)
    print(f"Ensuring deltas are cached for {split_name} {convo_type}...")
    ensure_deltas_cached(conversations, split_name, convo_type, temp_dir, f"Caching {convo_type} deltas")
    
    print(f"Analyzing cumulative scores for {convo_type} conversations...")
    for convo_id, shots in enumerate(tqdm(conversations, desc=f"Analyzing {convo_type} conversations")):
        convo_key = stable_convo_key(shots, convo_type)
        filepath = os.path.join(temp_dir, f"{convo_type}_{convo_key}.npy")
        
        if not os.path.exists(filepath):
            print(f"Warning: Missing deltas for {filepath}. Skipping convo.")
            continue
            
        try:
            convo_deltas = np.load(filepath)
        except Exception as e:
            print(f"Warning: Could not load {filepath}. Skipping. Error: {e}")
            continue

        # Turn 1 (index 0) is the *first user prompt*. The *delta* doesn't happen until Turn 2.
        # So, Turn 1 in the plot has a score of 0.
        results.append({"ID": f"{convo_type}_{convo_key}", "Turn": 1, "Score": 0.0, "Type": convo_type})
        cumulative_score = 0.0

        # convo_deltas[0] is the delta from T1 -> T2.
        # So, the score at Turn 2 is the cumulative score *after* this first delta.
        for turn_offset, delta in enumerate(convo_deltas):
            turn_idx = turn_offset + 2 # (Turn 2, 3, ...)
            
            projection_score = np.dot(delta[best_layer_index], best_probe_weights)
            cumulative_score += projection_score
            
            results.append({"ID": f"{convo_type}_{convo_key}", "Turn": turn_idx, "Score": cumulative_score, "Type": convo_type})
            
    return results


def analyze_cumulative_from_cached_deltas(conversations, split_name, convo_type, temp_dir, probe_weights, layer_index):
    """
    Scores conversations from already cached deltas for one candidate layer.
    Does not regenerate activations; callers should cache deltas once first.
    """
    results = []
    if not conversations:
        return results

    for shots in conversations:
        convo_key = stable_convo_key(shots, convo_type)
        filepath = os.path.join(temp_dir, f"{convo_type}_{convo_key}.npy")

        if not os.path.exists(filepath):
            print(f"Warning: Missing deltas for {filepath}. Skipping convo.")
            continue

        try:
            convo_deltas = np.load(filepath)
        except Exception as e:
            print(f"Warning: Could not load {filepath}. Skipping. Error: {e}")
            continue

        results.append(
            {
                "ID": f"{convo_type}_{convo_key}",
                "Turn": 1,
                "Score": 0.0,
                "Type": convo_type,
                "Layer": int(layer_index),
            }
        )
        cumulative_score = 0.0
        for turn_offset, delta in enumerate(convo_deltas):
            turn_idx = turn_offset + 2
            projection_score = np.dot(delta[layer_index], probe_weights)
            cumulative_score += projection_score
            results.append(
                {
                    "ID": f"{convo_type}_{convo_key}",
                    "Turn": turn_idx,
                    "Score": cumulative_score,
                    "Type": convo_type,
                    "Layer": int(layer_index),
                }
            )

    return results

def compute_accuracy_by_turn(df_scores, threshold):
    """
    Returns a DataFrame with accuracy per turn for a given threshold.
    """
    df_acc = df_scores.copy()
    df_acc['y_true'] = (df_acc['Type'] == 'attack').astype(int)
    df_acc['y_pred'] = (df_acc['Score'] > float(threshold)).astype(int)
    accuracy_by_turn = df_acc.groupby('Turn').apply(
        lambda x: accuracy_score(x['y_true'], x['y_pred'])
    ).reset_index(name='Accuracy')
    return accuracy_by_turn

def compute_threshold_metric(df_scores, threshold, metric_mode):
    """
    Computes a scalar accuracy metric for a given threshold.
    """
    df_acc = df_scores.copy()
    y_true = (df_acc['Type'] == 'attack').astype(int).to_numpy()
    y_pred = (df_acc['Score'] > float(threshold)).astype(int).to_numpy()

    if metric_mode == "micro":
        return float(accuracy_score(y_true, y_pred))
    if metric_mode == "macro_turn":
        acc_by_turn = compute_accuracy_by_turn(df_scores, threshold)
        return float(acc_by_turn['Accuracy'].mean()) if not acc_by_turn.empty else float("nan")
    raise ValueError(f"Unknown metric_mode: {metric_mode}")


# --- 4.5. K-fold CV: train probe per fold, score held-out fold ---
print("\n--- 4.5. Running K-Fold Cross-Validation ---")
if len(attack_train_convos) == 0 or len(benign_train_convos) == 0:
    print("⚠️ No training conversations available. Skipping train-only CV.")
    cv_results_df = pd.DataFrame()
    fold_probe_summaries = []
else:
    train_all_convos = attack_train_convos + benign_train_convos
    y_train_convos = np.concatenate([np.ones(len(attack_train_convos)), np.zeros(len(benign_train_convos))]).astype(int)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []
    fold_probe_summaries = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_train_convos)), y_train_convos), start=1):
        print(f"\n=== Fold {fold_idx}/{N_SPLITS} ===")
        train_idx = train_idx.tolist()
        val_idx = val_idx.tolist()

        attack_train = [train_all_convos[i] for i in train_idx if y_train_convos[i] == 1]
        benign_train = [train_all_convos[i] for i in train_idx if y_train_convos[i] == 0]
        attack_val = [train_all_convos[i] for i in val_idx if y_train_convos[i] == 1]
        benign_val = [train_all_convos[i] for i in val_idx if y_train_convos[i] == 0]

        print(f"📊 Fold split: {len(attack_train)} attack & {len(benign_train)} benign train.")
        print(f"📊 Fold split: {len(attack_val)} attack & {len(benign_val)} benign val.")

        # --- Train probe on fold train set ---
        attack_deltas_train = get_or_generate_deltas(
            attack_train,
            f"cv_fold{fold_idx}_train",
            "attack",
            TEMP_DIR,
            f"Generating Attack Deltas (Fold {fold_idx} Train)",
        )
        benign_deltas_train = get_or_generate_deltas(
            benign_train,
            f"cv_fold{fold_idx}_train",
            "benign",
            TEMP_DIR,
            f"Generating Benign Deltas (Fold {fold_idx} Train)",
        )

        if len(attack_deltas_train) == 0 or len(benign_deltas_train) == 0:
            print("⚠️ No training deltas for this fold. Skipping fold.")
            continue

        X_probe_train = np.concatenate([attack_deltas_train, benign_deltas_train])
        y_probe_train = np.concatenate([np.ones(len(attack_deltas_train)), np.zeros(len(benign_deltas_train))])
        print(f"✅ Fold {fold_idx}: training dataset has {X_probe_train.shape[0]} total delta samples.")

        probe_weights_by_layer, probe_accuracies = train_all_layer_differential_probes(
            X_probe_train, y_probe_train, random_seed=RANDOM_SEED
        )
        max_train_accuracy = float(np.max(probe_accuracies))
        train_best_layer = int(np.where(np.isclose(np.array(probe_accuracies), max_train_accuracy))[0][-1])
        print(
            f"✅ Fold {fold_idx}: trained {len(probe_weights_by_layer)} layer candidates "
            f"(train-acc best layer = {train_best_layer}, train acc = {max_train_accuracy:.4f})"
        )
        fold_probe_summaries.append(
            {
                "Fold": fold_idx,
                "TrainBestLayer": train_best_layer,
                "TrainBestLayerTrainAcc": max_train_accuracy,
            }
        )

        # --- Cache held-out validation deltas once, then score every candidate layer ---
        ensure_deltas_cached(
            attack_val,
            f"cv_fold{fold_idx}_val",
            "attack",
            TEMP_DIR,
            f"Caching Attack Deltas (Fold {fold_idx} Val)",
        )
        ensure_deltas_cached(
            benign_val,
            f"cv_fold{fold_idx}_val",
            "benign",
            TEMP_DIR,
            f"Caching Benign Deltas (Fold {fold_idx} Val)",
        )

        for layer_index, probe_weights in enumerate(probe_weights_by_layer):
            attack_results = analyze_cumulative_from_cached_deltas(
                attack_val, f"cv_fold{fold_idx}_val", "attack", TEMP_DIR, probe_weights, layer_index
            )
            benign_results = analyze_cumulative_from_cached_deltas(
                benign_val, f"cv_fold{fold_idx}_val", "benign", TEMP_DIR, probe_weights, layer_index
            )

            if not (attack_results or benign_results):
                print(f"⚠️ No validation results for fold {fold_idx}, layer {layer_index}.")
                continue

            fold_df = pd.DataFrame(attack_results + benign_results)
            fold_df["Fold"] = fold_idx
            fold_df["Layer"] = int(layer_index)
            fold_df["LayerTrainAcc"] = float(probe_accuracies[layer_index])
            fold_results.append(fold_df)

    if fold_results:
        cv_results_df = pd.concat(fold_results, ignore_index=True)
        scores_csv_path = os.path.join(DF_DIR, f'trajectoryprobe_cumulative_drift_scores_traincv_{model_id_alt}_{RUN_SUFFIX}.csv')
        cv_results_df.to_csv(scores_csv_path, index=False)
        print(f"\n✅ Train-only CV scoring complete. Combined validation scores saved to {scores_csv_path}")

        fold_summary_df = pd.DataFrame(fold_probe_summaries)
        fold_summary_path = os.path.join(DF_DIR, f'trajectoryprobe_probe_fold_summary_traincv_{model_id_alt}_{RUN_SUFFIX}.csv')
        fold_summary_df.to_csv(fold_summary_path, index=False)
        print(f"✅ Fold probe summary saved to {fold_summary_path}")
    else:
        print("⚠️ No fold results were generated. Creating empty DataFrame.")
        cv_results_df = pd.DataFrame()

print("\n--- 4.55 Selecting Layer + Threshold by Train-Only CV ---")
selected_layer_index = None
best_threshold = 0.0
layer_threshold_ablation_df = pd.DataFrame()
if 'cv_results_df' not in locals() or cv_results_df.empty:
    print("⚠️ No CV results available. Cannot select layer/threshold by CV.")
else:
    ablation_rows = []
    for layer_index in sorted(cv_results_df["Layer"].unique().tolist()):
        layer_df = cv_results_df[cv_results_df["Layer"] == layer_index].copy()
        scores = layer_df["Score"].to_numpy(dtype=float)
        if len(scores) == 0:
            continue

        lo = float(np.quantile(scores, THRESHOLD_RANGE_LO_Q))
        hi = float(np.quantile(scores, THRESHOLD_RANGE_HI_Q))
        if np.isclose(lo, hi):
            lo, hi = lo - 1.0, hi + 1.0
        pad = (hi - lo) * THRESHOLD_RANGE_PAD_FRAC
        lo -= pad
        hi += pad

        thresholds = np.linspace(lo, hi, THRESHOLD_GRID_POINTS).astype(float)
        thresholds = np.unique(np.sort(np.concatenate([thresholds, np.array([0.0])])))
        fold_ids = sorted(layer_df["Fold"].unique().tolist())
        fold_dfs = {fid: layer_df[layer_df["Fold"] == fid].copy() for fid in fold_ids}

        for thr in thresholds:
            row = {"Layer": int(layer_index), "Threshold": float(thr)}
            fold_metrics = []
            for fid in fold_ids:
                metric = compute_threshold_metric(fold_dfs[fid], thr, metric_mode=THRESHOLD_SELECT_METRIC)
                fold_metrics.append(metric)
                row[f"Fold{fid}"] = metric
            row["Mean"] = float(np.nanmean(fold_metrics)) if fold_metrics else float("nan")
            row["Std"] = float(np.nanstd(fold_metrics)) if fold_metrics else float("nan")
            ablation_rows.append(row)

    if not ablation_rows:
        print("⚠️ No layer-threshold ablation rows were generated.")
    else:
        layer_threshold_ablation_df = pd.DataFrame(ablation_rows)
        layer_threshold_ablation_df = layer_threshold_ablation_df.sort_values(
            ["Mean", "Threshold", "Layer"], ascending=[False, True, True]
        )
        layer_threshold_ablation_df["AbsThr"] = layer_threshold_ablation_df["Threshold"].abs()
        best_mean = float(layer_threshold_ablation_df["Mean"].max())
        best_candidates = layer_threshold_ablation_df[
            np.isclose(layer_threshold_ablation_df["Mean"], best_mean, equal_nan=False)
        ].copy()
        best_candidates = best_candidates.sort_values(["AbsThr", "Threshold", "Layer"], ascending=[True, True, True])
        selected_layer_index = int(best_candidates.iloc[0]["Layer"])
        best_threshold = float(best_candidates.iloc[0]["Threshold"])

        layer_threshold_ablation_df = layer_threshold_ablation_df.drop(columns=["AbsThr"], errors="ignore")
        layer_threshold_ablation_path = os.path.join(
            DF_DIR, f"trajectoryprobe_layerthreshold_ablation_traincv_{model_id_alt}_{RUN_SUFFIX}.csv"
        )
        layer_threshold_ablation_df.to_csv(layer_threshold_ablation_path, index=False)
        print(f"✅ Layer-threshold ablation table saved to {layer_threshold_ablation_path}")
        print("\n--- Top Layer/Threshold Candidates (by CV mean) ---")
        print(layer_threshold_ablation_df.head(20).to_string(index=False))
        print(
            f"\n✅ Best layer+threshold selected (metric={THRESHOLD_SELECT_METRIC}): "
            f"layer={selected_layer_index}, threshold={best_threshold:.6f}, CV mean={best_mean:.4f}"
        )

        # Also save a compact per-layer best summary for easier inspection.
        per_layer_best = (
            layer_threshold_ablation_df.sort_values(["Layer", "Mean", "Threshold"], ascending=[True, False, True])
            .groupby("Layer", as_index=False)
            .head(1)
            .sort_values(["Mean", "Layer"], ascending=[False, True])
        )
        per_layer_best_path = os.path.join(
            DF_DIR, f"trajectoryprobe_layerthreshold_best_per_layer_traincv_{model_id_alt}_{RUN_SUFFIX}.csv"
        )
        per_layer_best.to_csv(per_layer_best_path, index=False)
        print(f"✅ Per-layer best threshold summary saved to {per_layer_best_path}")

print("\n--- 4.6 Training final probe on full train split ---")
if len(attack_train_convos) == 0 or len(benign_train_convos) == 0:
    print("⚠️ No training conversations available. Skipping final training/testing.")
    final_test_results_df = pd.DataFrame()
    final_best_layer_index = None
    final_best_probe_weights = None
else:
    attack_deltas_train_full = get_or_generate_deltas(
        attack_train_convos,
        "final_train",
        "attack",
        TEMP_DIR,
        "Generating Attack Deltas (Final Train)",
    )
    benign_deltas_train_full = get_or_generate_deltas(
        benign_train_convos,
        "final_train",
        "benign",
        TEMP_DIR,
        "Generating Benign Deltas (Final Train)",
    )

    if len(attack_deltas_train_full) == 0 or len(benign_deltas_train_full) == 0:
        print("⚠️ No training deltas for final training. Skipping.")
        final_test_results_df = pd.DataFrame()
        final_best_layer_index = None
        final_best_probe_weights = None
    else:
        X_probe_train_full = np.concatenate([attack_deltas_train_full, benign_deltas_train_full])
        y_probe_train_full = np.concatenate([np.ones(len(attack_deltas_train_full)), np.zeros(len(benign_deltas_train_full))])
        print(f"✅ Final training dataset has {X_probe_train_full.shape[0]} total delta samples.")

        final_probe_weights_by_layer, final_probe_accs = train_all_layer_differential_probes(
            X_probe_train_full, y_probe_train_full, random_seed=RANDOM_SEED
        )
        if selected_layer_index is None:
            print("⚠️ No CV-selected layer available; falling back to train-accuracy layer selection.")
            max_accuracy = float(np.max(final_probe_accs))
            candidate_indices = np.where(np.array(final_probe_accs) == max_accuracy)[0]
            final_best_layer_index = int(candidate_indices[-1])
        else:
            final_best_layer_index = int(selected_layer_index)
        final_best_probe_weights = final_probe_weights_by_layer[final_best_layer_index]
        print(
            f"✅ Final probe selected layer {final_best_layer_index} "
            f"(train acc = {final_probe_accs[final_best_layer_index]:.4f}, CV threshold = {best_threshold:.6f})"
        )

        final_checkpoint_path = os.path.join(
            DF_DIR, f"trajectoryprobe_final_probe_checkpoint_{model_id_alt}_{RUN_SUFFIX}.npz"
        )
        np.savez(
            final_checkpoint_path,
            best_layer_index=np.array(final_best_layer_index),
            best_threshold=np.array(best_threshold),
            best_probe_weights=final_best_probe_weights,
            final_probe_accs=np.array(final_probe_accs),
            selected_by=np.array("layerthreshold_cv"),
        )
        print(f"✅ Final probe checkpoint saved to {final_checkpoint_path}")

        print("\n--- 4.7 Scoring once on fixed final test split ---")
        attack_final_results = analyze_cumulative_from_cache(
            attack_test_convos, "final_test", "attack", TEMP_DIR, final_best_probe_weights, final_best_layer_index
        )
        benign_final_results = analyze_cumulative_from_cache(
            benign_test_convos, "final_test", "benign", TEMP_DIR, final_best_probe_weights, final_best_layer_index
        )
        final_test_results_df = pd.DataFrame(attack_final_results + benign_final_results)
        final_scores_path = os.path.join(DF_DIR, f"trajectoryprobe_cumulative_drift_scores_finaltest_{model_id_alt}_{RUN_SUFFIX}.csv")
        final_test_results_df.to_csv(final_scores_path, index=False)
        print(f"✅ Final test scores saved to {final_scores_path}")


# --- 5. Plotting and Visualization (Cumulative Trajectory) ---
print("\n--- 5. Plotting Cumulative Trajectory ---")
if 'final_test_results_df' in locals() and not final_test_results_df.empty:
    plt.figure(figsize=(14, 8))
    
    # The palette keys must be lowercase to match the data ("attack", "benign")
    sns.lineplot(
        data=final_test_results_df, x="Turn", y="Score", hue="Type", style="Type",
        markers=True, errorbar=('ci', 95), palette={"attack": "red", "benign": "blue"}
    )
    
    plt.axhline(0, color='grey', linestyle='--', label='Neutral Trajectory')
    plt.title('Cumulative Stateful Trajectory During Conversations (Final Test, 95% CI)', fontsize=16)
    plt.xlabel('Adversarial/Benign Turn Number')
    plt.ylabel(f'Cumulative Projection Score (Layer {final_best_layer_index})')
    plt.legend(title="Conversation Type")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set x-ticks to be integers
    max_turn_plot = int(final_test_results_df['Turn'].max())
    if max_turn_plot > 0:
        plt.xticks(
            ticks=range(1, max_turn_plot + 1),
            rotation=90
        )

    output_filename = os.path.join(FIGURE_DIR, f'cumulative_truly_stateful_differential_probe_FINALTEST_{model_id_alt}_{RUN_SUFFIX}.pdf')
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"\n📈 Trajectory plot saved to {output_filename}")
    plt.show()
else:
    print("\n⚠️ No results were generated. Skipping trajectory plot.")


# --- 6. Accuracy Analysis ---
print("\n--- 6. Calculating and Plotting Accuracy from Cumulative Score ---")

if 'cv_results_df' not in locals() or cv_results_df.empty:
    print("⚠️ 'cv_results_df' not found or is empty. Skipping threshold selection and final test accuracy.")
else:
    # --- 6.1 Layer-threshold CV summary ---
    print("\n--- 6.1 Layer-Threshold CV Summary ---")
    if selected_layer_index is None or layer_threshold_ablation_df.empty:
        print("⚠️ No layer-threshold CV selection available. Using fallback threshold/layer.")
    else:
        print(
            f"Using CV-selected layer={selected_layer_index}, threshold={best_threshold:.6f} "
            f"for final test accuracy."
        )
        selected_layer_ablation = layer_threshold_ablation_df[
            layer_threshold_ablation_df["Layer"] == selected_layer_index
        ].sort_values("Threshold")

        plt.figure(figsize=(12, 6))
        plt.plot(selected_layer_ablation["Threshold"], selected_layer_ablation["Mean"], label="CV mean")
        plt.fill_between(
            selected_layer_ablation["Threshold"].to_numpy(),
            (selected_layer_ablation["Mean"] - selected_layer_ablation["Std"]).to_numpy(),
            (selected_layer_ablation["Mean"] + selected_layer_ablation["Std"]).to_numpy(),
            alpha=0.2,
            label="±1 std",
        )
        plt.axvline(best_threshold, color="black", linestyle="--", label=f"best={best_threshold:.4g}")
        plt.title(f"Layer-Threshold Ablation (layer={selected_layer_index}, metric={THRESHOLD_SELECT_METRIC})")
        plt.xlabel("Threshold for Score > threshold -> Attack")
        plt.ylabel("Cross-validated accuracy")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        ablation_fig_path = os.path.join(
            FIGURE_DIR, f"layerthreshold_ablation_traincv_{model_id_alt}_{RUN_SUFFIX}.pdf"
        )
        plt.savefig(ablation_fig_path, format="pdf", bbox_inches="tight")
        print(f"📈 Layer-threshold ablation plot saved to {ablation_fig_path}")
        plt.show()

    # --- 6.2 Final test accuracy by turn using CV-selected threshold ---
    if 'final_test_results_df' not in locals() or final_test_results_df.empty:
        print("⚠️ 'final_test_results_df' not found or is empty. Skipping final test accuracy.")
        accuracy_by_turn = pd.DataFrame()
    else:
        accuracy_by_turn = compute_accuracy_by_turn(final_test_results_df, best_threshold)

        print("\n--- Final Test Accuracy at Each Turn (CV-Selected Threshold) ---")
        print(accuracy_by_turn.to_string(index=False))

        acc_csv_path = os.path.join(DF_DIR, f'trajectoryprobe_cumulative_drift_accs_finaltest_{model_id_alt}_{RUN_SUFFIX}.csv')
        accuracy_by_turn.to_csv(acc_csv_path, index=False)
        print(f"\n✅ Final test accuracies saved to {acc_csv_path}")
    
        # Plot the accuracy
        plt.figure(figsize=(14, 8))
        sns.lineplot(
            data=accuracy_by_turn,
            x='Turn',
            y='Accuracy',
            marker='o',
            color='b' # Blue for accuracy
        )
        
        plt.title(f'Differential Probe Accuracy Over Time (Final Test, thr={best_threshold:.4g})', fontsize=16)
        plt.xlabel('Conversation Turn Number')
        plt.ylabel('Test Accuracy')
    
        # Set x-ticks to be integers
        max_turn_plot = int(accuracy_by_turn['Turn'].max())
        if max_turn_plot > 0:
            plt.xticks(
                ticks=range(1, max_turn_plot + 1),
                rotation=90
            )
        
        plt.ylim(0.0, 1.05) # Accuracy is from 0.0 to 1.0
        plt.axhline(0.5, color='grey', linestyle='--', label='Random Chance (0.5)')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        # Save the figure
        acc_output_filename = os.path.join(FIGURE_DIR, f'cumulative_truly_stateful_differential_probe_ACCURACY_FINALTEST_{model_id_alt}_{RUN_SUFFIX}.pdf')
        plt.savefig(acc_output_filename, format='pdf', bbox_inches='tight')
        print(f"\n📈 Accuracy plot saved to {acc_output_filename}")
        plt.show()

print("\n--- Pipeline Complete ---")
