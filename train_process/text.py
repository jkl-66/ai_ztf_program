import os
import json
import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
from peft import PeftModel
from huggingface_hub import login

# Hugging Face认证（与train.py保持一致）
login(token=" ")

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
            prompt_parts.append(f"<|im_start|>user\n{user_q}<|im_end|>")
            prompt_parts.append(f"<|im_start|>assistant\n{assistant_a}<|im_end|>")

        # 添加当前问题
        prompt_parts.append(f"<|im_start|>user\n{question}<|im_end|>")
        prompt_parts.append(f"<|im_start|>assistant\n")

        return "".join(prompt_parts)
    except Exception as e:
        print(f"构造prompt失败: {e}")
        raise

# 3. 问答生成函数（优化参数适配LoRA模型）
def generate_answer(
    model, 
    tokenizer, 
    question: str, 
    history: list | None = None,
    max_new_tokens: int = 256,  # 与训练时的max_length保持一致
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_history: bool = True
):
    """
    生成问题回答
    - use_history: 是否使用对话历史（多轮问答）
    """
    try:
        # 构造prompt
        prompt = build_qa_prompt(question, history if (use_history and history) else None)

        # 配置生成参数（与train.py中的generate_text函数保持一致）
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config
            )

        # 解码结果并提取assistant回答部分
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 清理回答文本
        answer = answer.split("<|im_end|>")[0].strip()
        
        return answer
    except Exception as e:
        print(f"生成回答失败: {e}")
        raise

# 4. 问答交互函数（支持单轮/多轮）
def qa_interaction(model, tokenizer, use_mult_rounds: bool = False):
    """问答交互主函数"""
    print("=== Qwen LoRA 微调模型问答交互系统 ===")
    print("输入'退出'、'q'或'quit'结束交互")
    print("输入'清空'或'clear'清空对话历史")
    print(f"多轮对话模式: {'开启' if use_mult_rounds else '关闭'}")
    print("-" * 50)

    history: list[tuple[str, str]] = []

    while True:
        try:
            # 获取用户问题
            question = input("\n🤔 请输入你的问题: ")
            
            if question.lower() in ["退出", "q", "quit"]:
                print("👋 再见！")
                break
            elif question.lower() in ["清空", "clear"]:
                history.clear()
                print("🧹 对话历史已清空")
                continue
            elif not question.strip():
                print("❌ 请输入有效问题")
                continue

            print("🤖 正在思考...")
            
            # 生成回答
            answer = generate_answer(
                model, 
                tokenizer, 
                question, 
                history if use_mult_rounds else None,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9
            )

            # 显示回答
            print(f"\n🤖 模型回答:")
            print(f"{answer}")

            # 保存到历史（多轮模式）
            if use_mult_rounds:
                history.append((question, answer))
                print(f"\n📝 对话轮次: {len(history)}")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，再见！")
            break
        except Exception as e:
            print(f"❌ 交互过程中发生错误: {e}")
            print("请重试或输入'退出'结束程序")

# 5. 批量测试函数
def batch_test(model, tokenizer, test_questions: list[str]):
    """批量测试模型回答质量"""
    print("=== 批量测试模式 ===")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- 测试 {i}/{len(test_questions)} ---")
        print(f"问题: {question}")
        
        try:
            answer = generate_answer(model, tokenizer, question, max_new_tokens=200)
            print(f"回答: {answer}")
        except Exception as e:
            print(f"生成失败: {e}")
        
        print("-" * 50)

# 6. 主函数（整合模型加载与交互）
if __name__ == "__main__":
    try:
        # LoRA模型路径（与train.py中的output_dir一致）
        lora_model_path = "./qwen_7b_trained_with_prompt_lora"
        base_model_name = "Qwen/Qwen-7B-Chat"
        
        # 检查模型路径是否存在
        if not os.path.exists(lora_model_path):
            print(f"❌ 模型路径不存在: {lora_model_path}")
            print("请确保已完成模型训练，或检查路径是否正确")
            exit(1)

        print("🚀 开始加载LoRA微调模型...")
        # 加载模型
        model, tokenizer = load_trained_model(lora_model_path, base_model_name)
        
        print("\n✅ 模型加载完成！")
        
        # 选择交互模式
        print("\n请选择模式:")
        print("1. 单轮问答（默认）")
        print("2. 多轮对话")
        print("3. 批量测试")
        
        choice = input("\n请输入选择 (1/2/3): ").strip()
        
        if choice == "2":
            qa_interaction(model, tokenizer, use_mult_rounds=True)
        elif choice == "3":
            # 预设测试问题
            test_questions = [
                "请介绍一下人工智能的发展历史",
                "如何学习编程？",
                "模仿鲁迅的笔触，描述一下春天的景色",
                "解释一下什么是机器学习",
                "写一首关于友谊的诗"
            ]
            batch_test(model, tokenizer, test_questions)
        else:
            qa_interaction(model, tokenizer, use_mult_rounds=False)
            
    except Exception as e:
        print(f"❌ 主程序运行失败: {e}")
        print("请检查模型路径和依赖是否正确安装")