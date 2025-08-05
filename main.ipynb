# In[1]:
%%capture
!pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton
!pip install --no-deps cut_cross_entropy unsloth_zoo
!pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
!pip install --no-deps unsloth

# In[2]:
from unsloth import FastLanguageModel
import torch
from google.colab import userdata

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.2-3B",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,
    token=userdata.get('HF_ACCESS_TOKEN')
)

# In[3]:
tokenizer.clean_up_tokenization_spaces = False

# In[4]:
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

train_embeddings = True

if train_embeddings:
    target_modules = target_modules + ["lm_head"]

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=target_modules,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

!pip install datasets -q
empty_prompt = """
{kashmiri_text}
"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    kashmiri_text_samples = examples["kashmiri_text"]
    training_prompts = []
    for text in kashmiri_text_samples:
        training_prompt = empty_prompt.format(kashmiri_text=text) + EOS_TOKEN
        training_prompts.append(training_prompt)
    return {"text": training_prompts}

from datasets import load_dataset
dataset = load_dataset("Aadilgani/kashmiri-text-dataset", split = 'train')
dataset = dataset.map(formatting_prompts_func, batched=True)

# In[6]:
for i, sample in enumerate(dataset):
    print(f"\n------ Sample {i + 1} ------")
    print(sample["text"])
    if i > 2:
        break

# In[7]:
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        max_steps=100,
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none"
    ),
)

# In[8]:
trainer_stats = trainer.train()

# In[9]:
from transformers import TextStreamer

def generate_text(model):
    FastLanguageModel.for_inference(model)
    inputs = tokenizer("", return_tensors="pt").to("cuda")
    text_streamer = TextStreamer(tokenizer)
    for token in model.generate(**inputs, streamer=text_streamer, max_new_tokens=100):
        print(token)

# In[10]:
for _ in range(3):
    generate_text(model)

model.push_to_hub_gguf(
    "Aadilgani/Llama-3.2-3B-kashmiri-lora-q4_k_m-GGUF",
    tokenizer,
    quantization_method="q4_k_m",
    token=userdata.get('HF_ACCESS_TOKEN')
)
# Step 2: Download GGUF model from Hugging Face using huggingface_hub
from huggingface_hub import hf_hub_download

gguf_path = hf_hub_download(
    repo_id="Aadilgani/Llama-3.2-3B-kashmiri-lora-q4_k_m-GGUF",
    filename="unsloth.Q4_K_M.gguf",  # Use exact name from your repo
    token=userdata.get("HF_ACCESS_TOKEN")  # Set to True if your repo is private and you’re logged in
)

# Step 3: Load the model using llama-cpp
from llama_cpp import Llama

llm = Llama(
    model_path=gguf_path,
    n_ctx=2048,
    n_threads=4
)

# Step 4: Generate response
def generate_response(prompt):
    output = llm(prompt, max_tokens=100)
    return output['choices'][0]['text']

# Interactive loop
print("Starting chat with the model. Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    print("Model:", generate_response(user_input))
