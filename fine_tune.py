import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
import os
from transformers import BitsAndBytesConfig

# Definir la ruta donde se guardarán los checkpoints en Google Drive
checkpoint_dir = "/content/drive/MyDrive/local-ai-code-assistant/models/finetuned_model"

# Cargar el dataset tokenizado desde Google Drive
dataset = load_from_disk("/content/drive/MyDrive/local-ai-code-assistant/tokenized_dataset_colab")

# Nombre del modelo a utilizar
model_name = "meta-llama/Llama-3.2-1B-Instruct"

# Cargar el tokenizador
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
tokenizer.pad_token = tokenizer.eos_token  # Usar el token de fin de secuencia como padding

# Función de preprocesamiento
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["func_documentation_string"],  
        text_pair=examples["func_code_string"],  
        padding="max_length",
        truncation=True,
        max_length=512
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# Aplicar preprocesamiento
dataset = dataset.map(preprocess_function, batched=True)

# Configuración para cargar el modelo en 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Cargar el modelo en GPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,  
    device_map="auto"  
)

# Configuración de LoRA
config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Aplicar LoRA al modelo
model = get_peft_model(model, config)

# Configurar los argumentos del entrenamiento
training_args = TrainingArguments(
    output_dir=checkpoint_dir,  
    save_strategy="steps",
    save_steps=100,  
    evaluation_strategy="steps",
    eval_steps=100,
    save_total_limit=3,  
    per_device_train_batch_size=2,  
    per_device_eval_batch_size=2,
    logging_steps=50,
    learning_rate=2e-4,
    weight_decay=0.01,
    num_train_epochs=1,
    fp16=True,  
    optim="adamw_torch"
)

# Buscar el último checkpoint en Google Drive (si existe)
if os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
    last_checkpoint = max([os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir)], key=os.path.getmtime)
    print(f"🔄 Reanudando desde el checkpoint: {last_checkpoint}")
else:
    last_checkpoint = None  # Asegurar que no intente reanudar si no hay un checkpoint válido

# Configurar el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer
)

# Iniciar Fine-Tuning (desde el último checkpoint si existe)
if last_checkpoint:
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("⚠️ No se encontró un checkpoint válido. Iniciando entrenamiento desde cero.")
    trainer.train()  

# Guardar el modelo final en Google Drive
trainer.save_model(checkpoint_dir)
print(f"✅ Fine-tuning completado y modelo guardado en {checkpoint_dir}")
