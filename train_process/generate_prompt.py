import json
import re
import random
from tqdm import tqdm

def load_and_clean_text(filepath: str) -> list[str]:
    """
    加载文本，进行清洗并按段落切分。
    
    Args:
        filepath: 文本文件的路径。

    Returns:
        一个由段落字符串组成的列表。
    """
    print(f"正在从 '{filepath}' 加载文本...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"错误：找不到文件 {filepath}")
        return []
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return []

    # 基础清洗：替换掉中文作品中常见的全角空格段首缩进。
    text = text.replace('　', '').replace(' ', '')
    
    # 按段落切分：使用一个或多个换行符作为分隔符。
    paragraphs = re.split(r'\n\s*\n+', text)
    
    # 过滤掉空段落或过短的段落（如仅有标点）。
    cleaned_paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 10]
    
    print(f"文本加载并清洗完毕，共获得 {len(cleaned_paragraphs)} 个有效段落。")
    return cleaned_paragraphs

def generate_prompts(
    paragraphs: list[str], 
    author_name: str,
    prompt_type: str,
    context_size: int = 1, 
    response_size: int = 1) -> list[dict]:
    """
    根据指定的类型（文章创作或问答）生成伪指令。

    Args:
        paragraphs: 清洗后的段落列表。
        author_name: 作者的名字（主要用于文学创作，问答时可忽略）。
        prompt_type: 指令类型，'article_creation' 或 'qa_answering'。
        context_size: 作为上下文的段落数量。
        response_size: 作为回答的段落数量。

    Returns:
        SFT数据列表。
    """
    print(f"开始生成 {prompt_type} 类型的伪指令...")
    sft_data = []
    
    if prompt_type == 'article_creation':
        templates = {
            "continuation": [
                "续写下面的故事或文章：\n\n{context}",
                "基于以下内容，撰写一段连贯的后续：\n\n{context}",
                "请根据下面的段落，继续创作一篇完整的文章：\n\n{context}"
            ],
            "style_imitation": [
                "请用'{author}'的风格，为以下段落写一段后续：\n\n{context}",
                "模仿作家'{author}'的笔触，对下面的主题进行扩展：\n\n{context}",
                "如果'{author}'来撰写接下来的内容，他会怎么写？\n\n{context}"
            ],
            "elaboration_and_expansion": [
                "请对以下主题进行更详细的阐述和扩展：\n\n{context}",
                "以下是一段描述，请为其注入更多细节和深入的分析：\n\n{context}",
                "请基于以下信息，展开论述并提供更多相关内容：\n\n{context}"
            ],
            "summary_and_creation": [
                "总结以下段落的核心思想，并在此基础上创作一篇短文：\n\n{context}",
                "'{context}'\n\n这段文字的主要观点是什么？请围绕此观点撰写一篇分析性文章。"
            ]
        }
    elif prompt_type == 'qa_answering':
        qa_templates = [
            "请根据以下内容回答问题：{context}\n\n问题：{question}",
            "阅读下面的文本，并回答相关问题：{context}\n\n问题：{question}",
            "基于以下信息，请提供关于'{question_topic}'的答案：{context}\n\n问题：{question}"
        ]
        
        def generate_question_from_context(context_text: str) -> str:
            sentences = re.split(r'[。！？\n]', context_text)
            sentences = [s.strip() for s in sentences if s.strip()]
            if sentences:
                chosen_sentence = random.choice(sentences)
                if len(chosen_sentence) > 10:
                    question_starters = ["请问", "关于", "什么是", "如何", "为什么"]
                    if not any(chosen_sentence.startswith(qs) for qs in question_starters):
                         return random.choice(question_starters) + " " + chosen_sentence.split('，')[0] + "？"
                    return chosen_sentence + "？"
            
            general_questions = [
                "这段文字主要讲了什么？",
                "请总结这段内容。",
                "这段文本的关键信息是什么？"
            ]
            return random.choice(general_questions)
        
        for i in tqdm(range(len(paragraphs) - response_size + 1), desc="生成QA Prompt"):
            output_text = "\n\n".join(paragraphs[i:i + response_size])
            question = generate_question_from_context(output_text)
            
            chosen_qa_template = random.choice(qa_templates)
            instruction = chosen_qa_template.format(context="", question=question, question_topic=question.replace("请问", "").replace("？", "").strip())
            
            sft_data.append({
                "instruction": instruction,
                "input": "", 
                "output": output_text
            })
        print(f"生成完毕，共创建了 {len(sft_data)} 条 {prompt_type} 伪指令数据。")
        return sft_data

    else:
        raise ValueError("无效的prompt_type。请选择 'article_creation' 或 'qa_answering'。")

    all_templates = [tpl for category in templates.values() for tpl in category]

    num_possible_prompts = len(paragraphs) - context_size - response_size + 1
    for i in tqdm(range(num_possible_prompts), desc="生成Prompt"):
        context_end = i + context_size
        response_end = context_end + response_size
        
        context = "\n\n".join(paragraphs[i:context_end])
        output = "\n\n".join(paragraphs[context_end:response_end])
        
        chosen_template = random.choice(all_templates)
        instruction = chosen_template.format(context=context, author=author_name)
        
        sft_data.append({
            "instruction": instruction,
            "input": "", 
            "output": output
        })
        
    print(f"生成完毕，共创建了 {len(sft_data)} 条 {prompt_type} 伪指令数据。")
    return sft_data

def save_to_jsonl(data: list[dict], filepath: str):
    """
    将数据以jsonl格式（每行一个JSON对象）保存。
    """
    print(f"正在将数据写入到 '{filepath}'...")
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in tqdm(data, desc="保存中"):
            json_record = json.dumps(entry, ensure_ascii=False)
            f.write(json_record + '\n')
    print("文件保存成功！")

def main():
    # 直接在代码中定义路径和参数
    INPUT_FILE = "train_process/data.txt"
    OUTPUT_FILE = "train_process/output.jsonl"
    AUTHOR_NAME = "这位作家"
    CONTEXT_SIZE = 1
    RESPONSE_SIZE = 1
    PROMPT_TYPE = "article_creation"  # 或 "qa_answering"
    
    # 执行主流程
    paragraphs = load_and_clean_text(INPUT_FILE)
    if paragraphs:
        sft_dataset = generate_prompts(
            paragraphs,
            AUTHOR_NAME,
            PROMPT_TYPE,
            CONTEXT_SIZE,
            RESPONSE_SIZE
        )
        if sft_dataset:
            save_to_jsonl(sft_dataset, OUTPUT_FILE)

if __name__ == "__main__":
    main()