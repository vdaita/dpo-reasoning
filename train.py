from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import Adam
from fire import Fire
import json

from math_judge import MathJudge
from online_dpo_trainer_with_samples import OnlineDPOTrainerWithSamples
from trl.trl.trainer.online_dpo_trainer import OnlineDPOTrainer
from trl.trl.trainer.online_dpo_config import OnlineDPOConfig
from trl.trl.trainer.callbacks import LogCompletionsCallback
from trl.trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from peft import LoraConfig

model_name = "RUC-AIBOX/STILL-3-1.5B-preview"
dataset_name = "RUC-AIBOX/STILL-3-Preview-RL-Data"
eval_dataset_name = "Maxwell-Jia/AIME_2024"

lora_config = LoraConfig(
    r=256,
    lora_alpha=128,
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

def train_model(epochs=2, num_samples_per_prompt=8):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, peft_config=lora_config)

    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token
    else:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    ds = load_dataset(dataset_name, split="train")

    ds = ds.map(lambda x: {**x, "answer": f"${x['answer']}$"})
    ds = ds.map(lambda x: {**x, "prompt": [{"role": "user", "content": x['question']}]})
    ds = ds[:4000]
    print(json.dumps(ds[:3], indent=4))

    eval_ds = load_dataset(eval_dataset_name, split="train")
    eval_ds = eval_ds.map(lambda x: {**x, "prompt": [{"role": "user", "content": x['Problem']}]})
    eval_ds = eval_ds.map(lambda x: {**x, "Answer": f"${x['Answer']}$"})

    qa_dict = dict(zip(ds["question"] + eval_ds["Problem"], ds["answer"] + eval_ds["Answer"]))

    training_args = OnlineDPOConfig(
        output_dir="outputs/STILL-3-1.5B-preview-OnlineDPOSamples",
        logging_steps=10,
        num_train_epochs=epochs,
        fp16=True,
        fp16_backend="amp",
        use_vllm=True
    )

    trainer = OnlineDPOTrainerWithSamples(
        model=model,
        judge=MathJudge(qa_dict=qa_dict),
        args=training_args,
        train_dataset=ds,
        eval_dataset=eval_ds,
        num_samples=num_samples_per_prompt,
        processing_class=tokenizer
    )
    completions_callback = LogCompletionsCallback(
        trainer, num_prompts=8
    )
    trainer.add_callback(completions_callback)
    trainer.train()

if __name__ == "__main__":
    Fire(train_model)