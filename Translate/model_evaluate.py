import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np
import evaluate
import torch
from tqdm.auto import tqdm  # 导入tqdm库

print("Lift off!!")


# 1. 加载数据
print("Datasets")
full_dataset = load_from_disk("validation_dataset")
subset_size = len(full_dataset) // 16
dataset_validation = full_dataset.select(range(subset_size))
print("Go!!\n")

# 2. 加载模型和tokenizer
print("Model")
tokenizer = AutoTokenizer.from_pretrained(model_path)
if "mbart" in model_path:
    print("支持的语言代码:", list(tokenizer.lang_code_to_id.keys()))
    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = "zh_CN"
    assert "en_XX" in tokenizer.lang_code_to_id, f"无效的语言代码，可选: {list(tokenizer.lang_code_to_id.keys())}"
#如果使用的是T5预训练模型的checkpoints，需要对特殊的前缀进行检查
if model_path in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
    prefix = "translate English to Chinese: "
else:
    prefix = ""
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
print("Go!!\n")

# 3. 数据预处理（添加进度条）
max_input_length = 128
max_target_length = 128
source_lang = "en"
target_lang = "zh"

def preprocess_function(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Validation data preprocession")
tokenized_validation = dataset_validation.map(
    preprocess_function,
    batched=True,
    desc="Preprocessing"  # 添加进度条描述
)
print("Go!!\n")

# 4. 加载评估指标
print("Metrics")
bleu_metric = evaluate.load("bleu", trust_remote_code=True)
rouge_metric = evaluate.load("rouge", trust_remote_code=True)
print("Go!!\n")

# 5. 定义评估函数
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(preds, labels):
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    
    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    return {
        "bleu": round(bleu_result["bleu"], 4),
        "rouge1": round(rouge_result["rouge1"], 4),
        "rouge2": round(rouge_result["rouge2"], 4),
        "rougeL": round(rouge_result["rougeL"], 4)
    }

# 6. 进行预测（添加进度条）
print("Generating predictions:")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
model.to(device)

val_inputs = tokenized_validation["input_ids"]
val_labels = tokenized_validation["labels"]

batch_size = 32
all_preds = []
num_batches = (len(val_inputs) + batch_size - 1) // batch_size

# 使用tqdm包装循环
for i in tqdm(range(0, len(val_inputs), batch_size), 
            total=num_batches,
            desc="Generating predictions",
            unit="batch"):
    batch = val_inputs[i:i+batch_size]
    batch = torch.tensor(batch).to(device)
    
    with torch.no_grad():
        generated = model.generate(
            input_ids=batch,
            max_length=max_target_length,
            num_beams=4
        )
    all_preds.extend(generated.cpu().numpy())

print("✓ Predictions generated\n")

# 7. 计算指标
print("Computing metrics...")
metrics = compute_metrics(all_preds, val_labels)


# 8. 打印结果（美化输出）
print("\n" + "="*50)
print("Final Evaluation Results:".center(50))
print("="*50)
print(f"{'BLEU Score:':<15}{metrics['bleu']:>10}")
print(f"{'ROUGE-1:':<15}{metrics['rouge1']:>10}")
print(f"{'ROUGE-2:':<15}{metrics['rouge2']:>10}")
print(f"{'ROUGE-L:':<15}{metrics['rougeL']:>10}")
print("="*50 + "\n")

# 9. 样本对比（添加进度条）
print("Sample Predictions:")
sample_indices = range(min(3, len(val_inputs)))  # 确保不超过样本总数

for i in tqdm(sample_indices, desc="Showing samples", unit="sample"):
    print(f"\n[Sample {i+1}]")
    print(f"{'Input:':<10}{tokenizer.decode(val_inputs[i], skip_special_tokens=True)}")
    print(f"{'Reference:':<10}{tokenizer.decode(val_labels[i], skip_special_tokens=True)}")
    print(f"{'Prediction:':<10}{tokenizer.decode(all_preds[i], skip_special_tokens=True)}")