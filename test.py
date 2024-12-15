import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = './gpt2'

# 离线加载，仍使用 trust_remote_code=True，让其从本地目录读取自定义代码
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True
)

model.eval()

input_text = "你好，请介绍一下你自己。"
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)
output = model.generate(input_ids, max_new_tokens=50)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)