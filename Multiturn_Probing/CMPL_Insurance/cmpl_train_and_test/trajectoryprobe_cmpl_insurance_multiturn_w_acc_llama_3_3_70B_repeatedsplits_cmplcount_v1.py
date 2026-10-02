from pathlib import Path

from cmpl_repeated_split_count_runner import run_count_experiment


run_count_experiment(
    Path(__file__).with_name(
        "trajectoryprobe_cmpl_insurance_multiturn_w_acc_llama_3_3_70B_"
        "kfoldcrossval_cmpl_train_cmpltest.py"
    )
)
