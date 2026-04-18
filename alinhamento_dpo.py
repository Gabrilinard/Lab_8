
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
