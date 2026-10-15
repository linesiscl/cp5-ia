# Treinando LLM - CP5/CP6 (Generative AI)

Fine-tuning de uma LLM local com GPU para responder perguntas sobre a FIAP e conceitos de IA.

> Checkpoint 6 — Generative AI

## Integrantes do grupo

| Nome | RM |
| --- | --- |
| Aline Fernandes Zeppelini | RM97966 |
| Camilly Breitbach Ishida | RM551474 |
| Julia Leite Galvão | RM550201 |

A documentação completa, com prints e tabelas com os resultados obtidos durante o treinamento, estão no arquivo `resultados.docx`

## Modelo e técnica

- **Modelo base:** TinyLlama-1.1B-Chat-v1.0
- **Técnica:** fine-tuning com QLoRA (quantização 4-bit + LoRA)
- **Dataset:** `dataset/qa_fiap.jsonl` (30 pares instrução/resposta em português)
- **Bibliotecas:** transformers, peft, bitsandbytes, trl, accelerate


## Como rodar o treinamento

1. Abrir `notebook_colab.ipynb` no Google Colab
2. Ambiente de execução > Alterar tipo > GPU T4
3. Rodar as células em ordem
4. Upload de `dataset/qa_fiap.jsonl` quando pedido

## Como rodar o frontend

```bash
pip install -r requirements.txt
python app.py
```

## Stack

Python, PyTorch, Transformers, PEFT (LoRA), BitsAndBytes, TRL, Gradio, Google Colab.
