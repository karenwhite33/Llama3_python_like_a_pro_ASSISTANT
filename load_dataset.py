from datasets import load_dataset

# Cargar el dataset desde Hugging Face permitiendo código remoto
dataset = load_dataset("code-search-net/code_search_net", trust_remote_code=True)

# Ver la estructura del dataset
print(dataset)
