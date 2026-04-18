
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

BASE_PESOS = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ARQUIVO_PARES = Path("/content/dados/preferencias_hhh.jsonl")
DESTINO_MODELO = "/content/saidas/guardiao_alinhado"
PENALIDADE_DESVIO = 0.1
AMOSTRAS_POR_PASSO = 1
ACUMULACAO = 1
RODADAS = 6
COMPRIMENTO_MAXIMO = 512
RITMO_APRENDIZADO = 1e-4

def abrir_pares_de_preferencia(arquivo):
    linhas = []
    with open(arquivo, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if linha:
                linhas.append(json.loads(linha))
    return Dataset.from_list(linhas)

def esquema_de_quantizacao():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

def esquema_de_adaptacao():
    return LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

def montar_dupla_de_modelos(base):
    decodificador = AutoTokenizer.from_pretrained(base)
    if decodificador.pad_token is None:
        decodificador.pad_token = decodificador.eos_token
    quant = esquema_de_quantizacao()
    candidato = AutoModelForCausalLM.from_pretrained(base, quantization_config=quant, device_map="auto")
    candidato = prepare_model_for_kbit_training(candidato)
    candidato = get_peft_model(candidato, esquema_de_adaptacao())
    ancora = AutoModelForCausalLM.from_pretrained(base, quantization_config=quant, device_map="auto")
    return decodificador, candidato, ancora

def parametros_de_otimizacao():
    return DPOConfig(
        beta=PENALIDADE_DESVIO,
        output_dir=DESTINO_MODELO,
        per_device_train_batch_size=AMOSTRAS_POR_PASSO,
        gradient_accumulation_steps=ACUMULACAO,
        num_train_epochs=RODADAS,
        learning_rate=RITMO_APRENDIZADO,
        optim="paged_adamw_32bit",
        fp16=False,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        max_length=COMPRIMENTO_MAXIMO,
        remove_unused_columns=False,
        report_to="none",
    )

conjunto = abrir_pares_de_preferencia(ARQUIVO_PARES)
decodificador, candidato, ancora = montar_dupla_de_modelos(BASE_PESOS)

ciclo_dpo = DPOTrainer(
    model=candidato,
    ref_model=ancora,
    args=parametros_de_otimizacao(),
    train_dataset=conjunto,
    processing_class=decodificador,
)

historico = ciclo_dpo.train()
print(f"loss final: {historico.training_loss:.4f}")
ciclo_dpo.save_model(DESTINO_MODELO)
