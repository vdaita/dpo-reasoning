from torchtune.rlhf.loss import DPOLoss
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import Adam
from fire import Fire

from math_judge import MathJudge
from online_dpo_trainer_with_samples import OnlineDPOTrainerWithSamples
from trl.trl.trainer.online_dpo_config import OnlineDPOConfig
from trl.trl.trainer.callbacks import LogCompletionsCallback

model_name = "RUC-AIBOX/STILL-3-1.5B-preview"
dataset_name = "RUC-AIBOX/STILL-3-Preview-RL-Data"
eval_dataset_name = "Maxwell-Jia/AIME_2024"

def train_model(epochs=2, bsz=64, num_samples_per_prompt=8):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.eos_token = tokenizer.pad_token

    ds = load_dataset(dataset_name, split="train")

    ds = ds.map(lambda x: {**x, "answer": f"${x['answer']}$"})
    ds = ds.map(lambda x: {**x, "prompt": [{"role": "user", "content": x['question']}]})
    ds = ds.shuffle(seed=42)
    train_ds = ds[:4000]

    eval_ds = load_dataset(eval_dataset_name, split="train")
    eval_ds = eval_ds.map(lambda x: {**x, "prompt": [{"role": "user", "content": x['Problem']}]})
    eval_ds = eval_ds.map(lambda x: {**x, "Answer": f"${x['Answer']}$"})

    qa_dict = dict(zip(train_ds["question"] + eval_ds["Problem"], train_ds["answer"] + eval_ds["Answer"]))

    training_args = OnlineDPOConfig(
        output_dir="STILL-3-1.5B-preview-OnlineDPO",
        logging_steps=10
    )

    trainer = OnlineDPOTrainerWithSamples(
        model=model,
        judge=MathJudge(qa_dict=qa_dict),
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        num_samples=num_samples_per_prompt
    )
    completions_callback = LogCompletionsCallback(
        trainer, num_prompts=8
    )
    trainer.add_callback(completions_callback)
    trainer.train()

if __name__ == "__main__":
    Fire(train_model)