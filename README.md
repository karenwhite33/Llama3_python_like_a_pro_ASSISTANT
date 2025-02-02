# **Build a Local/Online AI Code Assistant with LLaMA 3B and QLoRA INT4**

## **The Problem**

AI-powered coding assistants have revolutionized software development, enhancing speed and efficiency. However, most require cloud-based processing, raising concerns about data privacy and security. Every function, variable, and line of code you input is sent to an external server—posing a major risk for industries handling proprietary, confidential, or sensitive data, such as finance, healthcare, and cybersecurity.

***The challenge is clear:***

How can developers leverage AI-driven coding assistance without compromising data privacy?

This project presents a local-first AI coding assistant that runs entirely on your own machine, keeping all processing private while still benefiting from AI-powered automation.

# **The Solution: Phyton Like A Pro**

**Core Technologies**
To build this AI coding assistant, Im going to use:

LLaMA 3B – A lightweight, open-source model for code generation.

QLoRA INT4 – Efficient fine-tuning that reduces memory consumption by 4x, enabling performance on low-resource environments.

Hugging Face & Gradio – A simple, interactive web interface for developers, deployed on Hugging Face Spaces with an offline download option for privacy.

Google Colab – For fine-tuning and experimentation.

This setup ensures all computations remain local, with an optional web deployment that doesn’t require cloud uploads.

# **How the Assistant Works**

My AI-powered coding assistant:

1️⃣ Understands natural language prompts to generate relevant Python code.

2️⃣ Runs locally, eliminating data leakage risks.

3️⃣ Uses Gradio for an intuitive web-based interface, while also allowing offline use to protect proprietary code.

![challenges_solutions_table](https://github.com/user-attachments/assets/fcec1985-8398-4f30-a327-ccfc90cbc7e4)



--------------------------------------------------

Important

Fine-tuned LLaMA 3B for Python code generation using QLoRA INT4, reducing memory usage by 4x.

Achieved 92% syntax correctness in generated code.

Reduced token generation latency by 18%, improving efficiency for real-time coding.

challenges_solutions_table.png

# **ONLINE**

🔵 Huggingface Models: https://huggingface.co/karenwhiteg/python-like-a-pro

🔴 Huggingface Spaces: https://huggingface.co/spaces/karenwhiteg/Python_Like_A_Pro

📗Huggingface Dataset: https://huggingface.co/datasets/karenwhiteg/Python_Like_A_Pro


# **LOCAL** 
Carpeta Drive: https://drive.google.com/file/d/11dBJpETqHf6t3QOaR4I4tq65VSODYhaO/view?usp=sharing

🚀 Instalación y Uso del Asistente de Código

1️⃣ Descarga y Extrae los Archivos

2️⃣ Instala Python (si no lo tienes)

3️⃣ Crea un Entorno Virtual (Opcional, Recomendado)

4️⃣ Instala Requirements

5️⃣ Ejecuta el Asistente

6️⃣ ¡Listo! El Asistente se Abrirá en tu Navegador

