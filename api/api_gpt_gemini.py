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
PROMPT_DIR = os.path.join(root, "lyx/Chinese_datasets_api/prompt")
RESULTS_DIR = os.path.join(root, "lyx/results/Chinese_datasets_api")

MAX_RETRIES = 5
INITIAL_DELAY = 5
REQUEST_INTERVAL = 1

# 可用的数据集
DATA_FILES = {
    "toxic": os.path.join(root, "lyx/Chinese_datasets_api/data/Toxic_data.json"),
    "test": os.path.join(root, "lyx/Chinese_datasets_api/data/test.json")
}

# 支持的模型列表
AVAILABLE_MODELS = [
    "gpt-4o-2024-11-20",
    "o1-2024-12-17",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro-preview-05-06"
]

def call_model(prompt, content, model_name):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": content}
    ]
    
    retry_delay = INITIAL_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            response = openai.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0
            )
            
            # 针对Gemini模型输出调试信息
            if model_name in ["gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-preview-05-06"]:
                print(f"\n[DEBUG - {model_name}] 原始响应结构:")
                print(response)
            
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e):  
                print(f"超出速率限制，{retry_delay}秒后重试 (尝试 {attempt + 1}/{MAX_RETRIES})")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"错误: {str(e)}")
                return f"错误: {str(e)}"
    
    return "错误_超过最大重试次数"

def process_data(model_names, dataset_name):
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    today = datetime.date.today()
    
    # 加载选定的数据集
    input_file = DATA_FILES[dataset_name]
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"加载了 {len(data)} 条数据，来自数据集: {dataset_name}")
    
    # 缓存所有数据集的prompt
    dataset_prompts = {}
    datasets = set(item["dataset"] for item in data)
    for dataset in datasets:
        prompt_path = os.path.join(PROMPT_DIR, f"{dataset}_prompt.txt")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            dataset_prompts[dataset] = f.read().strip()
    
    # 为每个模型处理数据
    for model_name in model_names:
        print(f"\n开始使用模型 {model_name} 处理数据集 {dataset_name}")
        output_file = os.path.join(RESULTS_DIR, f"{model_name}-{dataset_name}-{today}.json")
        
        results = []
        for item in tqdm(data, desc=f"正在使用 {model_name} 处理"):
            toxic = item["toxic"]
            dataset = item["dataset"]
            content = USER_TEMPLATE.format(sentence=toxic.strip())
            
            response = call_model(dataset_prompts[dataset], content, model_name)
            results.append({
                "toxic": toxic,
                "rewritten": response.strip(),
                "dataset": dataset
            })
            time.sleep(REQUEST_INTERVAL)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"结果已保存到 {output_file}")
    
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用指定模型处理有害文本并进行改写')
    parser.add_argument('--models', type=str, nargs='+', default=["gpt-4o-2024-11-20"], 
                        help='使用的模型名称，可以指定多个模型，用空格分隔')
    parser.add_argument('--all', action='store_true', help='使用所有支持的模型')
    parser.add_argument('--dataset', type=str, choices=['toxic', 'test'], default='toxic',
                        help='选择处理的数据集：toxic (Toxic_data.json) 或 test (test.json)')
    args = parser.parse_args()
    
    # 确定要使用的模型列表
    if args.all:
        models_to_use = AVAILABLE_MODELS
    else:
        models_to_use = args.models
        
    print(f"将使用以下模型处理数据: {', '.join(models_to_use)}")
    print(f"选择的数据集: {args.dataset}")
    process_data(models_to_use, args.dataset)
