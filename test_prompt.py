# Definir el mensaje del sistema
system_message = (
    "You are a Senior Software Developer and a helpful AI programming assistant. "
    "Follow the user's requirements carefully and generate high-quality Python code. "
    "1. Think step-by-step: First, describe your plan in pseudocode. "
    "2. Generate code: Write the Python code in a single block. "
    "3. Minimize prose: Keep explanations concise. "
    "4. Always respond in English."
)

# Función para crear el prompt estructurado
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

# Función para aplicar el formato al dataset
def create_prompt(sample):
    instruction = sample["func_documentation_string"]  # Corrección de clave
    code = sample.get("func_code_string", "")  # Usar .get() para evitar errores si no hay respuesta
    
    return prompt_template(instruction, code)

# 🔹 Prueba con un ejemplo del dataset
dataset = {
    "train": [
        {
            "func_documentation_string": "Write a Python function to calculate the sum of elements in a list.",
            "func_code_string": ""
        }
    ]
}

# Crear el prompt de prueba
prompt = create_prompt(dataset["train"][0])

print("Generated Prompt:")
print(prompt)
