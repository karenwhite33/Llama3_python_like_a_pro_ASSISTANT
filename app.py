import warnings
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Evitar warnings
warnings.filterwarnings("ignore")

# Path al modelo local
model_path = "D:/AI Bootcamp Github/local-ai-code-assistant/models/finetuned_model"

# Cargar tokenizer y modelo conINT4 quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
) if torch.cuda.is_available() else None

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    quantization_config=bnb_config,
    device_map="auto"
)

# Optimizar modelo con torch.compile() (PyTorch 2.0+ users)
if hasattr(torch, "compile"):
    model = torch.compile(model)

# Funcion para generar codigo
def generate_code(instruction):
    
    prompt = f"<|startoftext|>\n### Instruction:\nYou are a Senior Software Developer and a helpful AI programming assistant. Generate only Python code without explanations. Respond only in English.\n\n### Task:\n{instruction}\n\n### Response:\n"
    
    encoded_input = tokenizer(prompt, return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **encoded_input, 
        max_new_tokens=512,
        do_sample=True,
        temperature=0.3,
        top_k=50,
        top_p=0.9,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id  
    )

    decoded_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    # Limpiar output
    generated_code = decoded_output[0].replace(prompt, "").strip()
    generated_code = generated_code.replace("<|endoftext|>", "").strip()
    
    return generated_code

# Crar la Gradio interface
interface = gr.Interface(
    fn=generate_code,
    inputs=gr.Textbox(lines=2, placeholder="How can I convert minutes into seconds?..."),
    outputs=gr.Textbox(lines=10, placeholder="Generated Python code will appear here."),
    title="Python like a Pro",
    description="Need a Python function? Just describe it here:"
)

# Launch app con public link
if __name__ == "__main__":
    interface.launch(share=True)
