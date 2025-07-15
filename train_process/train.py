import os
import json
import random
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.model_selection import train_test_split
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import Trainer, DataCollatorForLanguageModeling
from tqdm import tqdm
from transformers.training_args import TrainingArguments
from datasets import Dataset, DatasetDict, load_dataset # <--- 导入 DatasetDict 和 load_dataset
from transformers.generation.configuration_utils import GenerationConfig
import torch
import torch.nn.functional as F
from huggingface_hub import login
from transformers.trainer_callback import TrainerCallback
import gc  # 用于垃圾回收
# LoRA相关导入
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from peft.utils import prepare_model_for_kbit_training
# DeepSeek API相关导入
import requests
import time
import asyncio
import aiohttp

# Hugging Face认证 - 如果网络有问题可以注释掉
# login(token="")

# DeepSeek API配置
class DeepSeekConfig:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.api_key = " "
        self.base_url = base_url
        self.model_name = "deepseek-chat"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

# 数据处理和伪prompt生成类
class DataProcessor:
    def __init__(self, deepseek_config: DeepSeekConfig):
        self.deepseek_config = deepseek_config
    
    def load_raw_text(self, file_path: str) -> str:
        """加载原始文本数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def split_text_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list:
        """将长文本分割成适合处理的块"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # 尝试在句号处分割，避免截断句子
            if end < len(text):
                last_period = chunk.rfind('。')
                if last_period > chunk_size // 2:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 20]
    
    def generate_pseudo_prompts(self, chunks: list, enable_qa: bool = True, enable_article: bool = True) -> list:
        """为文本块生成伪prompt，支持问答和文章生成两种类型"""
        
        # 问答类型的prompt模板
        qa_prompts = [
            "根据以下文本内容，提出一个问题并回答：",
            "请基于以下文本，生成3个相关的问答对：",
            "阅读以下文本，然后回答：这段文字的主要观点是什么？",
            "根据以下内容，解释其中提到的重要概念：",
            "请分析以下文本中的因果关系：",
            "基于以下文本，总结作者的主要论点：",
            "请解释以下文本中的关键信息：",
            "根据以下内容，回答：作者想要表达什么？"
        ]
        
        # 文章生成类型的prompt模板
        article_prompts = [
            "请续写以下文本，保持相同的写作风格：",
            "模仿以下文本的风格，写一段相关内容：",
            "请改写以下文本，使其更加生动有趣：",
            "基于以下文本的主题，创作一段新的内容：",
            "请扩展以下文本，增加更多细节描述：",
            "模仿以下文本的语言特色，写一段类似的文字：",
            "请对以下文本进行创意改编：",
            "基于以下文本的情感基调，创作相关内容："
        ]
        
        # 组合可用的prompt类型
        available_prompts = []
        if enable_qa:
            available_prompts.extend(qa_prompts)
        if enable_article:
            available_prompts.extend(article_prompts)
        
        if not available_prompts:
            raise ValueError("至少需要启用一种prompt类型（问答或文章生成）")
        
        pseudo_prompts = []
        for i, chunk in enumerate(chunks):
            prompt_type = available_prompts[i % len(available_prompts)]
            
            # 确定prompt类别
            category = "问答" if prompt_type in qa_prompts else "文章生成"
            
            pseudo_prompts.append({
                "instruction": f"{prompt_type}\n\n{chunk}",
                "input": "",
                "chunk_id": i,
                "category": category
            })
        
        return pseudo_prompts
    
    def generate_teacher_responses(self, pseudo_prompts: list) -> list:
        """使用DeepSeek API生成教师模型回答"""
        responses = []
        
        for i, prompt_data in enumerate(tqdm(pseudo_prompts, desc="生成教师模型回答")):
            instruction = prompt_data["instruction"]
            response = call_deepseek_api(self.deepseek_config, instruction)
            
            if response.strip():
                responses.append({
                    "instruction": instruction,
                    "input": prompt_data["input"],
                    "output": response,
                    "chunk_id": prompt_data["chunk_id"]
                })
            
            time.sleep(0.1)  # 避免API限流
        
        return responses
    
    def save_to_jsonl(self, data: list, output_path: str):
        """保存数据到JSONL文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"数据已保存到: {output_path}")
    
    def process_raw_data_to_jsonl(self, input_file: str, output_file: str, chunk_size: int = 500, 
                                 enable_qa: bool = True, enable_article: bool = True):
        """完整的数据处理流程：从原始文本到JSONL训练数据"""
        print(f"开始处理原始数据文件: {input_file}")
        print(f"启用功能 - 问答: {enable_qa}, 文章生成: {enable_article}")
        
        # 1. 加载原始文本
        raw_text = self.load_raw_text(input_file)
        print(f"原始文本长度: {len(raw_text)} 字符")
        
        # 2. 分割文本
        chunks = self.split_text_into_chunks(raw_text, chunk_size=chunk_size)
        print(f"文本已分割为 {len(chunks)} 个块")
        
        # 3. 生成伪prompt
        pseudo_prompts = self.generate_pseudo_prompts(chunks, enable_qa=enable_qa, enable_article=enable_article)
        print(f"生成了 {len(pseudo_prompts)} 个伪prompt")
        
        # 统计prompt类型分布
        qa_count = sum(1 for p in pseudo_prompts if p['category'] == '问答')
        article_count = sum(1 for p in pseudo_prompts if p['category'] == '文章生成')
        print(f"  - 问答类型: {qa_count} 个")
        print(f"  - 文章生成类型: {article_count} 个")
        
        # 4. 使用教师模型生成回答
        training_data = self.generate_teacher_responses(pseudo_prompts)
        print(f"成功生成 {len(training_data)} 条训练数据")
        
        # 5. 保存到JSONL文件
        self.save_to_jsonl(training_data, output_file)
        
        return training_data

# DeepSeek API调用函数
def call_deepseek_api(config: DeepSeekConfig, prompt: str, max_tokens: int = 1000) -> str:
    """
    调用DeepSeek API生成教师模型的回答
    """
    try:
        url = f"{config.base_url}/v1/chat/completions"
        data = {
            "model": config.model_name,
            "messages": [
                {"role": "system", "content": "你是一个优秀的中文写作助手，请根据用户的要求生成高质量的文本。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        response = requests.post(url, headers=config.headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            print(f"DeepSeek API调用失败: {response.status_code}, {response.text}")
            return ""
    except Exception as e:
        print(f"DeepSeek API调用异常: {e}")
        return ""

# 批量调用DeepSeek API生成教师数据
def generate_teacher_data(config: DeepSeekConfig, prompts: List[str], batch_size: int = 5) -> List[str]:
    """
    批量调用DeepSeek API生成教师模型的回答
    """
    teacher_responses = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="生成教师模型数据"):
        batch_prompts = prompts[i:i + batch_size]
        
        for prompt in batch_prompts:
            response = call_deepseek_api(config, prompt)
            teacher_responses.append(response)
            time.sleep(0.1)  # 避免API限流
    
    return teacher_responses

# GPU内存优化设置
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # 清理GPU缓存
    # 设置内存分配策略
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# 设置随机种子确保结果可复现
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# 1. 加载JSONL伪prompt数据并分割为三部分
def load_and_split_data(jsonl_path: str, test_size: float = 0.1, val_size: float = 0.1) -> DatasetDict:
    print(f"加载JSONL数据: {jsonl_path}")
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    total_size = len(data)
    if total_size == 0:
        raise ValueError("JSONL文件中未找到有效数据")

    # 按8:1:1比例分割
    train_data, temp_data = train_test_split(data, test_size=(test_size + val_size), random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=test_size/(test_size+val_size), random_state=42)

    # 返回一个DatasetDict对象
    return DatasetDict({
        'train': Dataset.from_list(train_data),
        'val': Dataset.from_list(val_data),
        'test': Dataset.from_list(test_data)
    })

# 2. 数据预处理函数
def preprocess_function(examples, tokenizer, max_length=1024):
    """
    数据预处理函数，将instruction和output格式化为Qwen的对话格式
    """
    prompts = []
    for instruction, output in zip(examples['instruction'], examples['output']):
        prompt = f"<|im_start|>user\n{instruction}<|im_end|><|im_start|>assistant\n{output}<|im_end|>"
        prompts.append(prompt)
    
    # 分词
    tokenized = tokenizer(
        prompts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # 设置labels（用于计算损失）
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

# 知识蒸馏Trainer类
class DistillationTrainer(Trainer):
    def __init__(self, teacher_outputs: Optional[List[str]] = None, temperature: float = 3.0, alpha: float = 0.7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_outputs = teacher_outputs or []
        self.temperature = temperature
        self.alpha = alpha
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        计算知识蒸馏损失
        """
        # 获取学生模型的输出
        outputs = model(**inputs)
        student_logits = outputs.logits
        
        # 计算标准的语言模型损失
        if "labels" in inputs:
            labels = inputs["labels"]
            # 移动labels到正确的设备
            labels = labels.to(student_logits.device)
            
            # 计算交叉熵损失
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            lm_loss = outputs.loss if hasattr(outputs, 'loss') else torch.tensor(0.0, device=student_logits.device)
        
        # 如果有教师输出，计算蒸馏损失
        if self.teacher_outputs and len(self.teacher_outputs) > 0:
            # 这里简化处理，实际应用中可以更复杂
            # 由于我们使用的是文本形式的教师输出，这里主要使用标准损失
            total_loss = lm_loss
        else:
            total_loss = lm_loss
        
        return (total_loss, outputs) if return_outputs else total_loss

# 提前退出回调函数
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience: int = 3):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and isinstance(metrics, dict) and "eval_loss" in metrics:
            val_loss = metrics["eval_loss"]
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"验证损失连续 {self.patience} 次未降低，提前停止训练...")
                    control.should_training_stop = True

# 3. 完整的数据处理和训练流程
def train_model_from_raw_data(
    raw_data_path: str,
    deepseek_config: DeepSeekConfig,
    model_name: str = "Qwen/Qwen-7B-Chat",
    output_dir: str = "./qwen_7b_trained_with_deepseek_lora",
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    num_epochs: int = 15,
    max_length: int = 1024,
    chunk_size: int = 500,
    jsonl_output_path: str = "processed_training_data.jsonl",
    enable_qa: bool = True,
    enable_article: bool = True
):
    """
    完整的数据处理和训练流程：从原始文本到训练完成的模型
    
    Args:
        raw_data_path: 原始文本数据路径 (data.txt)
        deepseek_config: DeepSeek API配置
        model_name: 学生模型名称
        output_dir: 输出目录
        chunk_size: 文本分块大小
        jsonl_output_path: 处理后的JSONL文件保存路径
        enable_qa: 是否启用问答功能
        enable_article: 是否启用文章生成功能
    """
    print("=== 开始完整的数据处理和训练流程 ===")
    
    # 第一步：数据处理 - 将原始文本转换为训练数据
    print("\n第一步：数据处理")
    data_processor = DataProcessor(deepseek_config)
    
    # 处理原始数据生成JSONL训练文件
    training_data = data_processor.process_raw_data_to_jsonl(
        input_file=raw_data_path,
        output_file=jsonl_output_path,
        chunk_size=chunk_size,
        enable_qa=enable_qa,
        enable_article=enable_article
    )
    
    print(f"\n数据处理完成，生成了 {len(training_data)} 条训练样本")
    
    # 第二步：加载处理后的训练数据
    print("\n第二步：加载训练数据")
    datasets = load_and_split_data(jsonl_output_path)
    
    # 准备教师输出（用于知识蒸馏）
    teacher_outputs = [item['output'] for item in training_data]

    # 第三步：模型训练
    print("\n第三步：开始模型训练")
    print(f"训练数据集大小: {len(datasets['train'])}")
    print(f"验证数据集大小: {len(datasets['validation'])}")
    
    # 加载Qwen-7B-Chat模型和分词器
    print(f"加载Qwen模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_cache=False
    )

    # 设置pad_token
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    assert isinstance(im_end_id, int), "<|im_end|> token ID not found or not an integer."
    print(f"Using '<|im_end|>' (ID: {im_end_id}) as pad token.")
    tokenizer.pad_token_id = im_end_id
    model.config.pad_token_id = im_end_id
    tokenizer.padding_side = "right"
    
    # 移动模型到GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print(f"模型已移动到设备: {device}")
    else:
        print("CUDA不可用，使用CPU训练")
    
    # 配置LoRA参数
    print("配置LoRA参数...")
    
    # 自动查找Linear层作为target_modules
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # 只选择注意力和MLP相关的Linear层
            if any(keyword in name for keyword in ["attn", "mlp", "proj", "linear"]):
                # 提取模块名称的最后一部分
                module_name = name.split('.')[-1]
                if module_name not in target_modules:
                    target_modules.append(module_name)
    
    print(f"自动检测到的target_modules: {target_modules}")
    
    # 如果自动检测失败，使用通用配置
    if not target_modules:
        target_modules = ["c_attn", "c_proj"]
        print(f"使用默认target_modules: {target_modules}")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )
    
    # 应用LoRA到模型
    print("应用LoRA配置到模型...")
    model = get_peft_model(model, lora_config)
    
    # 确保LoRA模型在正确的设备上
    if torch.cuda.is_available():
        model = model.to(device)
        print(f"LoRA模型已移动到设备: {device}")
    
    # 打印可训练参数数量
    model.print_trainable_parameters()
    
    # 验证LoRA参数确实需要梯度
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
    print(f"需要梯度的参数数量: {len(trainable_params)}")
    if len(trainable_params) > 0:
        print(f"前5个可训练参数: {trainable_params[:5]}")
    else:
        print("警告：没有找到需要梯度的参数！")
    
    print(f"LoRA模型配置完成，大幅减少了可训练参数数量")

    # 手动预处理数据
    print("手动预处理数据...")

    def manual_tokenize(dataset, tokenizer, max_length):
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        
        pad_token_id = tokenizer.pad_token_id
        assert pad_token_id is not None, "Tokenizer's pad_token_id is not set!"

        for example in tqdm(dataset, desc="Tokenizing and Manually Padding"):
            if not isinstance(example, dict) or 'instruction' not in example or 'output' not in example:
                continue
            prompt = f"<|im_start|>user\n{example['instruction']}<|im_end|><|im_start|>assistant\n{example['output']}<|im_end|>"
            
            # 分词和截断
            tokenized = tokenizer(prompt, truncation=True, max_length=max_length)
            
            input_ids = tokenized['input_ids']
            
            # 手动创建labels和attention_mask
            labels = input_ids.copy()
            attention_mask = [1] * len(input_ids)

            # 手动填充
            padding_length = max_length - len(input_ids)

            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            labels = labels + ([-100] * padding_length)

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)

        return Dataset.from_dict({
            'input_ids': input_ids_list,
            'labels': labels_list,
            'attention_mask': attention_mask_list
        })

    train_dataset = manual_tokenize(datasets['train'], tokenizer, max_length)
    val_dataset = manual_tokenize(datasets['val'], tokenizer, max_length)

    # 配置训练参数
    print("配置训练参数...")

    steps_per_epoch = len(train_dataset) // (4 * 4)
    if steps_per_epoch == 0:
        steps_per_epoch = 1

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=2e-4,
        weight_decay=0.01,
        gradient_accumulation_steps=8,
        fp16=False,
        bf16=True,
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=steps_per_epoch,
        save_steps=steps_per_epoch,
        load_best_model_at_end=True,
        save_total_limit=2,
        push_to_hub=False,
        report_to="none",
        warmup_steps=100,
        save_safetensors=True,
    )

    # 初始化知识蒸馏Trainer
    print("初始化知识蒸馏Trainer...")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = DistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(patience=3)],
        teacher_outputs=teacher_outputs,
        temperature=3.0,
        alpha=0.7
    )

    # 清理内存并开始训练
    print("清理GPU内存...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    print("开始训练...")
    trainer.train()

    # 保存LoRA适配器
    print(f"保存LoRA适配器到 {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 合并LoRA权重到基础模型（可选）
    merge_and_save_full_model = True
    if merge_and_save_full_model:
        print("正在合并LoRA权重到基础模型...")
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        
        # 加载LoRA适配器并合并
        model_with_lora = PeftModel.from_pretrained(base_model, output_dir)
        merged_model = model_with_lora.merge_and_unload()
        
        # 保存合并后的完整模型
        merged_output_dir = output_dir + "_merged"
        print(f"保存合并后的完整模型到 {merged_output_dir}")
        merged_model.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)
        
        print(f"模型训练和合并完成!")
        print(f"LoRA适配器保存在: {output_dir}")
        print(f"完整合并模型保存在: {merged_output_dir}")
        
        return merged_model, tokenizer
    else:
        print("训练完成! LoRA适配器已保存至本地")
        return model, tokenizer

# 4. 模型生成示例函数
def generate_text(
    model,
    tokenizer,
    instruction: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9
):
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    formatted_prompt = f"<|im_start|>user\n{instruction}<|im_end|><|im_start|>assistant\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text

# 运行训练
if __name__ == "__main__":
    # 配置DeepSeek API
    deepseek_config = DeepSeekConfig(
        api_key="your_deepseek_api_key_here",  # 请替换为实际的API密钥
        base_url="https://api.deepseek.com/v1/chat/completions",
        model="deepseek-chat"
    )
    
    # 原始数据文件路径
    raw_data_path = "data.txt"  # 原始文本数据
    jsonl_output_path = "deepseek_generated_training_data.jsonl"  # 生成的训练数据

    # 完整的数据处理和训练流程
    model, tokenizer = train_model_from_raw_data(
        raw_data_path=raw_data_path,
        deepseek_config=deepseek_config,
        model_name="Qwen/Qwen-7B-Chat",
        output_dir="./qwen_7b_trained_with_deepseek_lora",
        batch_size=1,
        learning_rate=2e-4,
        num_epochs=15,
        max_length=256,
        chunk_size=500,
        jsonl_output_path=jsonl_output_path,
        enable_qa=True,        # 启用问答功能
        enable_article=True    # 启用文章生成功能
    )
    
    print("\n=== 训练完成 ===")
    print(f"原始数据: {raw_data_path}")
    print(f"生成的训练数据: {jsonl_output_path}")
    print(f"训练后的模型: ./qwen_7b_trained_with_deepseek_lora")

    sample_instruction = "模仿作家鲁迅的笔触，对下面的主题进行扩展：\n\n院子里的两棵树，一棵是枣树，另一棵也是枣树。"
    print("\n--- 生成示例文本 ---")
    generated = generate_text(
        model,
        tokenizer,
        sample_instruction,
        max_new_tokens=200
    )
    print(f"输入指令:\n{sample_instruction}\n")
    print(f"模型续写:\n{generated}")