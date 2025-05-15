import os
import json
import time
import datetime
import argparse
from pathlib import Path
from tqdm import tqdm
import openai

with open('/mnt/workspace/xintong/api_key.txt', 'r') as f:
    lines = f.readlines()

API_KEY = lines[0].strip()
BASE_URL = lines[1].strip()

openai.api_key = API_KEY
openai.base_url = BASE_URL

root = "/mnt/workspace/xintong/"
USER_TEMPLATE = "输入：{sentence}"
INPUT_FILE = os.path.join(root, "lyx/Chinese_datasets_api/data/Toxic_data.json")
PROMPT_DIR = os.path.join(root, "lyx/Chinese_datasets_api/prompt")
RESULTS_DIR = os.path.join(root, "lyx/results/Chinese_datasets_api")

MAX_RETRIES = 5
INITIAL_DELAY = 5
REQUEST_INTERVAL = 1


def call_model(prompt, content, model_name):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": content}
    ]
    
    retry_delay = INITIAL_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            # 创建聊天完成请求
            response = openai.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                stream=True,
            )
            
            reasoning_content = ""  # 定义完整思考过程
            answer_content = ""     # 定义完整回复
            is_answering = False   # 判断是否结束思考过程并开始回复

            for chunk in response:
                if not chunk.choices:  
                    continue  # 跳过无效数据
                delta = chunk.choices[0].delta

                # 记录思考过程
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    reasoning_content += delta.reasoning_content
                
                # 记录最终的回答内容
                if delta.content is not None:
                    # 进入回复阶段
                    if not is_answering:
                        is_answering = True
                    answer_content += delta.content
            
            return answer_content, reasoning_content
            
        except Exception as e:
            if "429" in str(e):
                print(f"超出速率限制，{retry_delay}秒后重试 (尝试 {attempt + 1}/{MAX_RETRIES})")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"错误: {str(e)}")
                return f"错误: {str(e)}", ""
    
    return "错误_超过最大重试次数", ""


def process_data(model_name):

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    today = datetime.date.today()
    output_file = os.path.join(RESULTS_DIR, f"{model_name}-{today}.json")
   
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"加载了 {len(data)} 条数据")

    results = []
    dataset_prompts = {}  # 缓存prompt

    for item in tqdm(data, desc=f"正在使用 {model_name} 处理"):
        toxic = item["toxic"]
        dataset = item["dataset"]

        if dataset not in dataset_prompts:
            prompt_path = os.path.join(PROMPT_DIR, f"{dataset}_prompt.txt")
            with open(prompt_path, 'r', encoding='utf-8') as f:
                dataset_prompts[dataset] = f.read().strip()

        content = USER_TEMPLATE.format(sentence=toxic.strip())
        rewritten, reason = call_model(dataset_prompts[dataset], content, model_name)

        results.append({
            "toxic": toxic,
            "reason": reason.strip() if reason else "", 
            "rewritten": rewritten.strip(),
            "dataset": dataset
        })

        time.sleep(REQUEST_INTERVAL)  

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"结果已保存到 {output_file}")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用指定模型处理有害文本并进行改写')
    parser.add_argument('--model', type=str, default="qwen3-235b-a22b", help='使用的模型名称')
    args = parser.parse_args()

    print(f"开始使用模型 {args.model} 处理数据")
    process_data(args.model)