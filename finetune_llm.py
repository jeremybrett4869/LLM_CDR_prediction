import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig, Trainer, pipeline
from peft import LoraConfig
from datasets import Dataset
from langchain.prompts.prompt import PromptTemplate
from trl import SFTTrainer
import pandas as pd

model_name = "/disk2/ysun/llm/vicuna-7b-v1.5" # PATH TO LOAD THE BASE MODEL
new_model = "/disk2/ysun/llm/finetune/vicuna-7b-s" # PATH TO SAVE THE FINETUNED MODEL

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# Load model
based_model = AutoModelForCausalLM.from_pretrained(model_name, 
  device_map="auto"
)
based_model.config.use_cache = False
based_model.config.pretraining_tp = 1


# Prepare Dataset
train_prompt_data = pd.read_csv("./data/sensitivity/train_prompt_data_simple.csv") # PATH TO LOAD THE TRAINING DATASET
print(f'Training Dataset Size: {len(train_prompt_data)}')
q_a_lst = [] 
for i in range(len(train_prompt_data)):
    question = train_prompt_data.loc[i, "prompt"].split("[Reasoning].")[1].strip().strip("?")
    answer = train_prompt_data.loc[i, "answer"]
    q_a_lst.append((question, answer))
prompt_template = PromptTemplate(
    input_variables=["instruction", "question", "answer"], template="{instruction}\n{question}{answer}"
)
instruction = "Decide in a single word reflecting the drug sensitivity of the drug on the cell line with given mutations."
prompt_data = [prompt_template.format(instruction=instruction, question=q, answer=a) for q, a in q_a_lst]
dataset = Dataset.from_dict({"text": prompt_data})


peft_params = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    logging_steps=1,
    learning_rate=2e-4,
    fp16=True
)

trainer = SFTTrainer(
    model=based_model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

trainer.train()

trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)