from datasets import load_from_disk, DatasetDict

# Cargar dataset tokenizado completo
dataset = load_from_disk("formatted_dataset")

# Filtrar el conjunto de entrenamiento a 10,000 muestras
train_dataset = dataset["train"].shuffle(seed=42).select(range(10_000))

# Ajustar validación y test al 10% de train
validation_dataset = dataset["validation"].shuffle(seed=42).select(range(1_000))
test_dataset = dataset["test"].shuffle(seed=42).select(range(1_000))

# Crear nuevo dataset balanceado
balanced_dataset = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset
})

# Guardar el dataset corregido
balanced_dataset.save_to_disk("formatted_dataset_10000")

print("✅ Dataset filtrado correctamente y guardado como 'formatted_dataset_10000'.")
