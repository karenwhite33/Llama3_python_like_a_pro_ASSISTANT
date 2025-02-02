from datasets import load_dataset

# Cargar el dataset
dataset = load_dataset("code-search-net/code_search_net", trust_remote_code=True)

# Seleccionar solo las columnas necesarias
columns_to_keep = ["func_documentation_string", "func_code_string"]

# Filtrar el dataset para solo mantener Python y eliminar nulos
def filter_and_clean(example):
    return example["language"] == "python" and example["func_documentation_string"] and example["func_code_string"]

filtered_dataset = dataset.filter(filter_and_clean)

# Mostrar algunas muestras del dataset procesado
print(filtered_dataset["train"][0])
print(filtered_dataset["train"][1])
print(f"Total muestras después del filtrado: {len(filtered_dataset['train'])}")

# Guardar el dataset procesado en un formato optimizado (parquet)
filtered_dataset.save_to_disk("processed_dataset")
