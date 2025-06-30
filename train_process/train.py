import os
import json
import random
from typing import Dict, List, Any
import numpy as np
from sklearn.model_selection import train_test_split
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import Trainer, DataCollatorForLanguageModeling
from tqdm import tqdm
from transformers.training_args import TrainingArguments
from datasets import Dataset, DatasetDict # <--- 导入 DatasetDict
from transformers.generation.configuration_utils import GenerationConfig
import torch
from huggingface_hub import login
from transformers.trainer_callback import TrainerCallback
import gc  # 用于垃圾回收
# LoRA相关导入
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from peft.utils import prepare_model_for_kbit_training

# Hugging Face认证 - 如果网络有问题可以注释掉
# login(token="")

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

# 3. 主训练函数
def train_model(
    jsonl_path: str,
    model_name: str = "Qwen/Qwen-7B-Chat",
    output_dir: str = "./qwen_7b_trained_with_prompt_lora",
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    num_epochs: int = 15,
    max_length: int = 1024
):
    """
    使用伪prompt生成的JSONL数据训练Qwen-7B-Chat模型
    """
    # 加载并分割数据
    print("加载并分割数据...")
    datasets = load_and_split_data(jsonl_path)

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

    # 初始化Trainer
    print("初始化Trainer...")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(patience=3)]
    )

    # 清理内存并开始训练
    print("清理GPU内存...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    print("开始训练...")
    trainer.train()

    # 保存模型到本地
    print(f"保存最终模型到 {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("训练完成! 模型已保存至本地")
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
    jsonl_data_path = "train_process/output.jsonl"  # 修改为正确的文件路径

    model, tokenizer = train_model(
        jsonl_path=jsonl_data_path,
        model_name="Qwen/Qwen-7B-Chat",
        output_dir="./qwen_7b_trained_with_prompt_lora",
        batch_size=1,
        learning_rate=2e-4,
        num_epochs=15,
        max_length=256
    )

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