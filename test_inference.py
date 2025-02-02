import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Asegurar que estamos usando CPU
device = "cpu"
print(f"✅ Modelo se ejecutará en: {device}")

# Cargar el modelo fine-tuneado
model_path = "models/finetuned_model"  # Ruta local del modelo fine-tuneado
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # Usar el token EOS como padding

# Mover el modelo a CPU
model.to(device)
print("✅ Modelo fine-tuneado cargado correctamente.")

# Prompt Template final (eusado en el fine-tuning)
def prompt_template(instruction):
    full_prompt = "<|startoftext|>\n"
    full_prompt += "### Instruction:\n"
    full_prompt += "You are a Senior Software Developer and a helpful AI programming assistant. "
    full_prompt += "Generate only Python code without explanations. Respond only in English.\n\n"
    full_prompt += f"### Task:\n{instruction}\n\n"
    full_prompt += "### Response:\n"
    return full_prompt

# Función de inferencia optimizada
def generate_code(prompt, model, tokenizer):
    encoded_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)

    generated_ids = model.generate(
        **encoded_input,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    # Limpiar la salida eliminando etiquetas adicionales
    generated_code = decoded_output[0].replace(prompt, "").strip()
    generated_code = generated_code.replace("<|endoftext|>", "").strip()

    return generated_code

# Lista de pruebas con diferentes instrucciones
test_instructions = [
    "Write a Python function to check if a number is even.",
    "Write a Python function to find the maximum number in a list.",
    "Write a Python function to compute the factorial of a number.",
    "Write a Python function to implement the quicksort algorithm.",
    "Write a Python function to calculate the greatest common divisor (GCD) of two numbers.",
    "Write a Python function to generate the Fibonacci sequence up to n terms."
]

# Ejecutar pruebas y mostrar resultados
for instruction in test_instructions:
    print("-" * 80)
    print(f"📝 Código generado para: {instruction}\n")
    
    prompt = prompt_template(instruction)
    response = generate_code(prompt, model, tokenizer)
    
    print(response)
    print("-" * 80)
