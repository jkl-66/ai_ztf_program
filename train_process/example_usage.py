#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek教师模型训练Qwen-7B的使用示例

这个脚本展示了如何使用修改后的train.py进行完整的数据处理和模型训练
"""

from train import DeepSeekConfig, train_model_from_raw_data
import os

def main():
    print("=== DeepSeek教师模型训练Qwen-7B示例 ===")
    
    # 1. 检查原始数据文件是否存在
    raw_data_path = "data.txt"
    if not os.path.exists(raw_data_path):
        print(f"错误：找不到原始数据文件 {raw_data_path}")
        print("请确保data.txt文件存在于当前目录")
        return
    
    # 2. 配置DeepSeek API
    print("\n配置DeepSeek API...")
    deepseek_config = DeepSeekConfig(
        api_key="your_deepseek_api_key_here",  # 请替换为实际的API密钥
        base_url="https://api.deepseek.com/v1/chat/completions",
        model="deepseek-chat"
    )
    
    # 检查API密钥是否已配置
    if deepseek_config.api_key == "your_deepseek_api_key_here":
        print("⚠️  警告：请先在代码中配置您的DeepSeek API密钥！")
        print("请修改 api_key 参数为您的实际API密钥")
        return
    
    # 3. 设置训练参数
    print("\n设置训练参数...")
    training_params = {
        "raw_data_path": raw_data_path,
        "deepseek_config": deepseek_config,
        "model_name": "Qwen/Qwen-7B-Chat",
        "output_dir": "./qwen_7b_trained_with_deepseek_lora",
        "batch_size": 1,                    # 小批次，适合GPU内存较小的情况
        "learning_rate": 2e-4,
        "num_epochs": 10,                   # 可根据需要调整
        "max_length": 256,                  # 序列长度，可根据GPU内存调整
        "chunk_size": 500,                  # 文本分块大小
        "jsonl_output_path": "deepseek_generated_training_data.jsonl",
        "enable_qa": True,                  # 启用问答功能
        "enable_article": True              # 启用文章生成功能
    }
    
    print(f"原始数据文件: {training_params['raw_data_path']}")
    print(f"输出目录: {training_params['output_dir']}")
    print(f"生成的训练数据: {training_params['jsonl_output_path']}")
    print(f"启用功能: 问答={training_params['enable_qa']}, 文章生成={training_params['enable_article']}")
    
    # 4. 开始训练
    print("\n开始训练流程...")
    try:
        model, tokenizer = train_model_from_raw_data(**training_params)
        
        print("\n🎉 训练完成！")
        print(f"✅ LoRA适配器已保存到: {training_params['output_dir']}")
        print(f"✅ 完整模型已保存到: {training_params['output_dir']}_merged")
        print(f"✅ 训练数据已保存到: {training_params['jsonl_output_path']}")
        
        # 5. 简单测试生成
        print("\n测试模型生成...")
        test_prompt = "请写一段关于人工智能的文字"
        
        # 这里可以添加模型测试代码
        print(f"测试提示: {test_prompt}")
        print("(实际生成需要加载模型进行推理)")
        
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {str(e)}")
        print("请检查：")
        print("1. DeepSeek API密钥是否正确")
        print("2. 网络连接是否正常")
        print("3. GPU内存是否足够")
        print("4. 依赖包是否正确安装")

def check_environment():
    """检查运行环境"""
    print("检查运行环境...")
    
    required_packages = [
        'torch', 'transformers', 'datasets', 'peft', 
        'requests', 'aiohttp', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} (未安装)")
    
    if missing_packages:
        print(f"\n请安装缺失的包: pip install {' '.join(missing_packages)}")
        return False
    
    return True

if __name__ == "__main__":
    # 检查环境
    if not check_environment():
        print("\n环境检查失败，请先安装必要的依赖包")
        exit(1)
    
    # 运行主程序
    main()
    
    print("\n=== 使用提示 ===")
    print("1. 确保data.txt包含您要训练的原始文本")
    print("2. 配置正确的DeepSeek API密钥")
    print("3. 根据GPU内存调整batch_size和max_length")
    print("4. 训练完成后可以使用生成的模型进行推理")
