%%capture
!pip install torch torchvision torchaudio transformers peft trl accelerate datasets bitsandbytes

from google.colab import userdata
from google.colab import drive
userdata.get('HF_TOKEN')
drive.mount('/content/drive')
hf_token = "your_token"

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

dataset = load_dataset("s4mjang/AM_QA_dataset_TF", split="train") #llama3.1용
dataset = dataset.train_test_split(test_size=0.1, seed=42)   # 80/20
train_ds, val_ds = dataset["train"], dataset["test"]

system = """
<|begin_of_text|>
<|start_header_id|>SYSTEM<|end_header_id|>
You are a helpful assistant for the competition authority. Find a Suspicious Pattern in the target code and explain in detail. (ex. 'Self Preferencing', 'Price Fixing', 'Manipulation of Randomized Item Logic', 'None')
<|eot_id|>
<|start_header_id|>USER<|end_header_id|>
### Instruction:
"""
outro = "\n### Response:<|eot_id|><|start_header_id|>ASSISTANT<|end_header_id|>\n"

def merge_ir(example):
    return {
        "text": (
            system
            + example["instruction"].strip()
            + outro
            + example["output"].strip()
            + "\n<|eot_id|>"
        )
    }

train_ds = train_ds.map(merge_ir, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(  merge_ir, remove_columns=val_ds.column_names)


MODEL = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # pad_warnings 방지

bnb_conf = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_enable_fp32_cpu_offload=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_conf,
    device_map="auto",
    trust_remote_code=True
)


from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


base_model = prepare_model_for_kbit_training(model)
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],                   
    lora_dropout=0.1,    
    bias="none",
    task_type="CAUSAL_LM",  # Causal LM 작업
)
model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()

training_args = SFTConfig(
    output_dir="your_path",
    per_device_train_batch_size=4,    
    gradient_accumulation_steps=8,     
    max_steps=100,                      
    learning_rate=1e-4,              
    lr_scheduler_type="cosine",         
    warmup_steps=20,                    
    max_grad_norm=1.0,                
    weight_decay=0.05,                
    gradient_checkpointing=True,        
    optim="adamw_8bit",                

    eval_strategy="steps",
    eval_steps=10,                   
    load_best_model_at_end=True,      
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataset_text_field="text",
    save_strategy="steps",
    save_steps=20,                      
    save_total_limit=4,                
    max_seq_length=512,                
    fp16=True,                         
    report_to=None,                
    logging_strategy="steps",
    logging_steps=10,                   
    logging_first_step=True,          
    packing=True,                     
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=training_args,
    peft_config=lora_cfg,
    processing_class=tokenizer,              # 토크나이저 지정
)

trainer.train()
