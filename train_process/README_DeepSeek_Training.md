# DeepSeek教师模型知识蒸馏训练指南

本文档介绍如何使用DeepSeek作为教师模型，从原始文本数据开始，通过完整的数据处理和知识蒸馏流程训练Qwen-7B学生模型。

## 功能特性

### 🎯 核心功能
- **原始数据处理**: 自动将data.txt原始文本转换为训练数据
- **智能伪prompt生成**: 支持问答和文章生成两种类型的训练任务
- **知识蒸馏训练**: 使用DeepSeek API作为教师模型指导Qwen-7B学习
- **文本智能分块**: 自动将长文本分割为适合处理的块，保持语义完整性
- **LoRA微调**: 高效的参数微调，减少计算资源需求
- **模型合并**: 训练完成后自动合并LoRA权重到基础模型
- **灵活配置**: 支持多种训练参数和功能开关

### 🔧 技术架构
- **教师模型**: DeepSeek API (deepseek-chat)
- **学生模型**: Qwen-7B-Chat
- **训练方法**: LoRA + 知识蒸馏
- **数据处理**: 自动数据增强和预处理

## 使用方法

### 1. 环境准备

确保已安装必要的依赖包：
```bash
pip install torch transformers datasets peft accelerate
pip install requests aiohttp tqdm
```

### 2. API配置

在 `train.py` 文件中配置DeepSeek API：

```python
deepseek_config = DeepSeekConfig(
    api_key="your_actual_deepseek_api_key",  # 替换为真实API密钥
    base_url="https://api.deepseek.com/v1/chat/completions",
    model="deepseek-chat"
)
```

### 3. 原始数据准备

准备原始文本数据文件 `data.txt`，可以是任何中文文本内容：
```
这是一段原始文本内容...
可以是小说、散文、新闻、学术文章等任何文本。
系统会自动将其转换为训练数据。
```

**注意**: 不再需要手动准备JSONL格式的训练数据，系统会自动处理！

### 4. 开始训练

运行训练脚本：
```bash
python train.py
```

### 5. 训练参数配置

可以调整以下关键参数：

```python
model, tokenizer = train_model_from_raw_data(
    raw_data_path="data.txt",                     # 原始文本数据路径
    deepseek_config=deepseek_config,               # DeepSeek API配置
    model_name="Qwen/Qwen-7B-Chat",               # 学生模型名称
    output_dir="./qwen_7b_trained_with_deepseek_lora",  # 输出目录
    batch_size=4,                                  # 批次大小
    learning_rate=5e-5,                           # 学习率
    num_epochs=10,                                # 训练轮数
    max_length=512,                               # 最大序列长度
    chunk_size=500,                               # 文本分块大小
    jsonl_output_path="training_data.jsonl",      # 生成的训练数据保存路径
    enable_qa=True,                               # 启用问答功能
    enable_article=True                           # 启用文章生成功能
)
```

## 训练流程说明

### 📝 第一步：数据处理流程
1. **原始文本加载**: 读取data.txt原始文本文件
2. **智能分块**: 将长文本分割为适合处理的块，保持语义完整性
3. **伪prompt生成**: 为每个文本块生成问答或文章生成类型的指令
4. **教师模型调用**: 使用DeepSeek API为每个伪prompt生成高质量回答
5. **JSONL保存**: 将处理后的数据保存为标准训练格式

### 🎓 第二步：知识蒸馏训练
1. **数据加载**: 加载处理后的JSONL训练数据
2. **模型初始化**: 加载Qwen-7B基础模型和LoRA配置
3. **蒸馏训练**: 使用DistillationTrainer进行知识蒸馏
4. **损失计算**: 结合标准语言模型损失和蒸馏损失
5. **参数更新**: 仅更新LoRA适配器参数

### 💾 第三步：模型保存
1. **LoRA保存**: 保存轻量级的LoRA适配器
2. **模型合并**: 将LoRA权重合并到基础模型
3. **完整保存**: 保存可直接使用的完整模型

### 🎯 支持的训练类型
- **问答功能**: 基于文本内容生成问答对，提升模型理解和回答能力
- **文章生成**: 学习文本风格和写作技巧，提升创作和续写能力

## 输出文件说明

训练完成后会生成以下文件：

```
data.txt                                # 原始输入文本
deepseek_generated_training_data.jsonl  # DeepSeek生成的训练数据

qwen_7b_trained_with_deepseek_lora/     # LoRA适配器目录
├── adapter_config.json                 # LoRA配置文件
├── adapter_model.bin                   # LoRA权重文件
├── tokenizer.json                      # 分词器文件
└── ...

qwen_7b_trained_with_deepseek_lora_merged/  # 合并后的完整模型
├── config.json                         # 模型配置
├── pytorch_model.bin                   # 完整模型权重
├── tokenizer.json                      # 分词器文件
└── ...
```

### 📊 生成的训练数据格式

`deepseek_generated_training_data.jsonl` 文件包含以下格式的数据：
```json
{"instruction": "根据以下文本内容，提出一个问题并回答：\n\n[原始文本块]", "input": "", "output": "[DeepSeek生成的回答]", "chunk_id": 0, "category": "问答"}
{"instruction": "请续写以下文本，保持相同的写作风格：\n\n[原始文本块]", "input": "", "output": "[DeepSeek生成的续写]", "chunk_id": 1, "category": "文章生成"}
```

## 使用训练后的模型

### 方法1: 使用LoRA适配器
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat")
# 加载LoRA适配器
model = PeftModel.from_pretrained(base_model, "./qwen_7b_trained_with_deepseek_lora")
tokenizer = AutoTokenizer.from_pretrained("./qwen_7b_trained_with_deepseek_lora")
```

### 方法2: 使用合并后的完整模型
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./qwen_7b_trained_with_deepseek_lora_merged")
tokenizer = AutoTokenizer.from_pretrained("./qwen_7b_trained_with_deepseek_lora_merged")
```

## 性能优化建议

### 🚀 训练优化
- **批次大小**: 根据GPU内存调整batch_size (推荐1-8)
- **学习率**: 建议使用2e-4到5e-5之间的学习率
- **序列长度**: 根据数据特点调整max_length
- **教师比例**: teacher_data_ratio建议在0.2-0.5之间

### 💡 资源管理
- **GPU内存**: 使用gradient_checkpointing减少内存占用
- **API调用**: 合理控制并发请求数量，避免超出API限制
- **数据缓存**: 可以预先生成教师数据并缓存，避免重复API调用

## 故障排除

### 常见问题

1. **API密钥错误**
   - 检查DeepSeek API密钥是否正确
   - 确认API账户有足够的调用额度

2. **内存不足**
   - 减小batch_size
   - 减小max_length
   - 启用gradient_checkpointing

3. **训练中断**
   - 检查数据格式是否正确
   - 确认模型路径可访问
   - 查看错误日志定位问题

4. **API调用失败**
   - 检查网络连接
   - 确认API服务状态
   - 适当增加重试间隔

### 调试技巧
- 使用小数据集进行测试
- 启用详细日志输出
- 监控GPU使用情况
- 检查生成的教师数据质量

## 注意事项

⚠️ **重要提醒**:
1. 请确保DeepSeek API密钥的安全性，不要泄露给他人
2. 注意API调用费用，合理控制teacher_data_ratio
3. 训练过程中请保持网络连接稳定
4. 建议在训练前备份原始数据
5. 大规模训练建议使用多GPU环境

## 技术支持

如遇到问题，请检查：
- Python环境和依赖包版本
- CUDA和PyTorch兼容性
- 数据格式和路径正确性
- API配置和网络连接

---

**祝您训练顺利！** 🎉