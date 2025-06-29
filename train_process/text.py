import os
import json
import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
from peft import PeftModel
from huggingface_hub import login

# Hugging Face认证（与train.py保持一致）
login(token="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")

# 1. 加载LoRA训练后的模型和分词器
def load_trained_model(model_path: str, base_model_name: str = "Qwen/Qwen-7B-Chat"):
    """从本地路径加载LoRA训练后的Qwen模型"""
    try:
        print(f"加载基础模型: {base_model_name}")
        # 首先加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        print(f"加载LoRA适配器从: {model_path}")
        # 加载LoRA适配器
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()  # 设置为评估模式
        print("LoRA模型加载成功！")
        return model, tokenizer
    except Exception as e:
        print(f"加载模型失败: {e}")
        raise

# 2. 构造Qwen问答格式的prompt（与train.py保持一致）
def build_qa_prompt(question: str, history: list | None = None) -> str:
    """
    构造符合Qwen对话格式的问答prompt
    - history: 多轮对话历史，格式为[(user_question, assistant_answer)]
    """
    if history is None:
        history = []

    try:
        # 使用与训练时相同的格式
        prompt_parts = []
        for user_q, assistant_a in history:
            prompt_parts.append(f"