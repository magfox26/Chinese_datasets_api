# Chinese_datasets_api

## 目录
1.api存放api代码
[api_gpt-gemini.py](https://github.com/magfox26/Chinese_datasets_api/blob/main/api/api_gpt_gemini.py)包括：gpt-4o-2024-11-20，o1-2024-12-17，gemini-2.5-flash-preview-04-17，gemini-2.5-pro-preview-05-06
[api_qwen-deepseek.py](https://github.com/magfox26/Chinese_datasets_api/blob/main/api/api_qwen_deepseek.py)包括：qwen-max-2025-01-25，qwen3-235b-a22b，deepseek-v3，deepseek-r1
2.data存放数据  
3.prompt存放各数据集的prompt   

## 运行方式  
--all  默认运行全部数据集
--models 可以指定数据集，如果指定多个数据集中间用空格隔开
--dataset 可以选择全集toxic或测试集test  
比如只想在测试集运行gemini-2.5-pro-preview-05-06，运行方式应该是：  
`python api_gpt-gemini.py --models gemini-2.5-pro-preview-05-06 --dataset test`

## 日志
### 2025年5月16日  
首先进行测试，用2个窗口分别运行 `python api_gpt-gemini.py --all --dataset test`和`python api_qwen-deepseek.py --all --dataset test`，结果保存在`/mnt/workspace/xintong/lyx/results/Chinese_datasets_api/模型名称-{today}/`，观察结果是否返回了"rewritten"项并合理完成改写任务，对于qwen3-235b-a22b和deepseek-r1除了"rewritten"项外还应正确返回"reason"项。  

1.如果结果都正确  
重新运行 `python api_gpt-gemini.py --all --dataset toxic`和`python api_qwen-deepseek.py --all --dataset toxic`  
2.如果某一个模型比如gemini-2.5-pro-preview-05-06出现错误  
在测试时[api_gpt-gemini.py](https://github.com/magfox26/Chinese_datasets_api/blob/main/api/api_gpt_gemini.py)的57-59行默认打印了gemini系列的原始响应结构，根据相应结构可能61行的return response.choices[0].message.content需要调整
