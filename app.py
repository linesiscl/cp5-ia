"""
Frontend do assistente FIAP (CP6 - Generative AI).

Carrega o modelo base TinyLlama + o adaptador LoRA treinado no notebook
(pasta "tinyllama-fiap-lora", baixada do Colab) e expõe uma interface
Gradio para o usuário digitar o prompt e ver a resposta do modelo.

Como usar:
    1. Treinar o modelo no notebook_colab.ipynb e baixar o .zip gerado.
    2. Descompactar o adaptador LoRA dentro desta pasta "frontend/",
       de forma que exista "frontend/tinyllama-fiap-lora/".
    3. pip install -r requirements.txt
    4. python app.py
"""

import os

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = os.path.join(os.path.dirname(__file__), "tinyllama-fiap-lora")

SYSTEM_PROMPT = (
    "Voce e um assistente educacional da FIAP. "
    "Responda em portugues de forma clara e objetiva."
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo em uso: {device}")

print("Carregando tokenizer e modelo base...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)

if os.path.isdir(LORA_PATH):
    print(f"Carregando adaptador LoRA de {LORA_PATH} ...")
    model = PeftModel.from_pretrained(model, LORA_PATH)
else:
    print(
        f"AVISO: adaptador LoRA nao encontrado em '{LORA_PATH}'. "
        "Rodando apenas com o modelo base (sem fine-tuning)."
    )

if device == "cpu":
    model = model.to(device)
model.eval()


def responder(pergunta: str, temperature: float, max_new_tokens: int) -> str:
    if not pergunta or not pergunta.strip():
        return "Digite uma pergunta."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": pergunta},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    resposta = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return resposta.strip()


with gr.Blocks(title="Assistente FIAP - CP6") as demo:
    gr.Markdown("# Assistente FIAP\nFaça uma pergunta sobre a FIAP ou conceitos de IA.")

    with gr.Row():
        with gr.Column():
            entrada = gr.Textbox(
                label="Seu prompt",
                placeholder="Ex: O que é fine-tuning de um modelo de linguagem?",
                lines=3,
            )
            temperatura = gr.Slider(
                minimum=0.1, maximum=1.5, value=0.7, step=0.1, label="Temperatura"
            )
            max_tokens = gr.Slider(
                minimum=32, maximum=512, value=200, step=16, label="Máx. tokens na resposta"
            )
            botao = gr.Button("Perguntar", variant="primary")
        with gr.Column():
            saida = gr.Textbox(label="Resposta do modelo", lines=10)

    botao.click(
        fn=responder,
        inputs=[entrada, temperatura, max_tokens],
        outputs=saida,
    )

if __name__ == "__main__":
    demo.launch()
