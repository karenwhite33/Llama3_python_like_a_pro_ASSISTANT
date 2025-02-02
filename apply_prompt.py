from datasets import load_from_disk, DatasetDict
import os

# Cargar el dataset procesado
dataset = load_from_disk("processed_dataset")

# Mensaje del sistema optimizado
system_message = (
    "You are a Senior Software Developer and a helpful AI programming assistant. "
    "Generate only Python code without explanations. Respond only in English."
)

# Función para construir el prompt estructurado (versión final)
def prompt_template(instruction, code=""):
    full_prompt = "<|startoftext|>\n"
    full_prompt += "### Instruction:\n"
    full_prompt += system_message + "\n\n"
    full_prompt += "### Task:\n"
    full_prompt += instruction + "\n\n"
    full_prompt += "### Response:\n"
    if code:
        full_prompt += code
    full_prompt += "\n<|endoftext|>"
    
    return full_prompt

# Aplicar el prompt estructurado al dataset
def format_dataset(example):
    instruction = example["func_documentation_string"]
    code = example.get("func_code_string", "")
    example["formatted_prompt"] = prompt_template(instruction, code)
    return example

formatted_dataset = dataset.map(format_dataset)

# Guardar el nuevo dataset con los prompts aplicados
output_dir = "formatted_dataset"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

formatted_dataset.save_to_disk(output_dir)

print(f"✅ Dataset con prompts aplicado y guardado en '{output_dir}'")
