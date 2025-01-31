from torchtune.rlhf.loss import DPOLoss
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import Adam
from fire import Fire

def train_model(model_name="", dataset_name="", question_col="", answer_col="", epochs=2, bsz=64, num_generations=16):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    ds = load_dataset("math_dataset", "train")
    ds = ds.map(batched=True, batch_size=bsz)



if __name__ == "__main__":
    Fire(train_model)