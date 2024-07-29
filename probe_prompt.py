import pandas as pd
import re
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 
from sklearn.preprocessing import StandardScaler
from utils.func import *
from utils.class_probe import Probe 

import warnings
warnings.filterwarnings('ignore')

torch.random.manual_seed(202407) 
model = AutoModelForCausalLM.from_pretrained( 
    "microsoft/Phi-3-mini-4k-instruct",  
    device_map="cuda",  
    torch_dtype="auto",  
    trust_remote_code=True, 
    output_hidden_states=True, 
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct") 

pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
) 

generation_args = { 
    "max_new_tokens": 30, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False, 
} 

def Phi3ChatCompletion(messages):
    output = pipe(messages, **generation_args) 
    return output[0]['generated_text']

##### Load datasets #####
splits = {'test': 'college_mathematics/test-00000-of-00001.parquet', 'validation': 'college_mathematics/validation-00000-of-00001.parquet', 'dev': 'college_mathematics/dev-00000-of-00001.parquet'}
Col_Math = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])

splits = {'test': 'business_ethics/test-00000-of-00001.parquet', 'validation': 'business_ethics/validation-00000-of-00001.parquet', 'dev': 'business_ethics/dev-00000-of-00001.parquet'}
Biz_Ethics = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])

splits = {'test': 'professional_law/test-00000-of-00001.parquet', 'validation': 'professional_law/validation-00000-of-00001.parquet', 'dev': 'professional_law/dev-00000-of-00001.parquet'}
Prf_Law = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])

##### Preprocessing #####
Col_Math = Col_Math.drop(columns=['subject'])
Col_Math['answer'] = Col_Math['answer'] + 1
Col_Math['gen_answer'] = None
Col_Math['confidence'] = None
Col_Math['correct'] = None

Biz_Ethics = Biz_Ethics.drop(columns=['subject'])
Biz_Ethics['answer'] = Biz_Ethics['answer'] + 1
Biz_Ethics['gen_answer'] = None
Biz_Ethics['confidence'] = None
Biz_Ethics['correct'] = None

Prf_Law = Prf_Law.drop(columns=['subject'])
Prf_Law['answer'] = Prf_Law['answer'] + 1
Prf_Law['gen_answer'] = None
Prf_Law['confidence'] = None
Prf_Law['correct'] = None

# Evaluate
hidden_sizes = [256, 128, 64]
probe = Probe(input_size=3072, hidden_sizes=hidden_sizes, output_size=1).cuda()
probe.load_state_dict(torch.load('probe_model.pth'))
probe.eval()
print("Probe model loaded from probe_model.pth")

def evaluate_with_probe(dataset, model, tokenizer, probe):
    model.eval()
    results = []
    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0], desc="Evaluating with Probe"):
        messages = [
            {"role": "system", "content": "Read the question, provide your answer and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.\nUse the following format to answer:\nAnswer and Confidence (0-100): [ONLY the {answer_number}; not a complete sentence or symbols], [Your confidence level, please only include the numerical number in the range of 0-100]% \nOnly the answer and confidence, don't give me the explanations."},
            {"role": "user", "content": f"Question: {row['question']}\nOptions: 1. {row['choices'][0]}\n2. {row['choices'][1]}\n3. {row['choices'][2]}\n4. {row['choices'][3]}"}
        ]
        with torch.no_grad():
            inputs = tokenizer(" ".join([msg['content'] for msg in messages]), return_tensors="pt").input_ids.cuda()
            outputs = model(inputs, output_hidden_states=True)
            last_layer_features = outputs.hidden_states[-1][:, -1, :].detach().float()  # Ensure float type
        
        gen_answer, confidence = extract_answer_and_confidence(Phi3ChatCompletion(messages))
        if gen_answer == 0 and confidence == 0: continue
        prob = probe(last_layer_features).squeeze()  # Squeeze to match the target size
        prob = prob.unsqueeze(0) if prob.dim() == 0 else prob  # Ensure prob is at least 1D
        results.append({
            'answer': row['answer'],
            'gen_answer': gen_answer,
            'confidence': confidence,
            'probe': prob.item(),
            'correct': 1 if gen_answer == row['answer'] else 0
        })
    
    return pd.DataFrame(results)

col_math = evaluate_with_probe(Col_Math, model, tokenizer, probe)
min_val = col_math['probe'].min()
max_val = col_math['probe'].max()
col_math['probe'] = col_math['probe'].apply(lambda x: scaled_probe(x, min_val, max_val))
col_math.to_csv('./probe_results/col_math.csv', index=False)
print(col_math)

prf_law = evaluate_with_probe(Prf_Law, model, tokenizer, probe)
min_val = prf_law['probe'].min()
max_val = prf_law['probe'].max()
prf_law['probe'] = prf_law['probe'].apply(lambda x: scaled_probe(x, min_val, max_val))
prf_law.to_csv('./probe_results/prf_law.csv', index=False)
print(prf_law)

biz_ethics = evaluate_with_probe(Biz_Ethics, model, tokenizer, probe)
min_val = biz_ethics['probe'].min()
max_val = biz_ethics['probe'].max()
biz_ethics['probe'] = biz_ethics['probe'].apply(lambda x: scaled_probe(x, min_val, max_val))
biz_ethics.to_csv('./probe_results/biz_ethics.csv', index=False)
print(biz_ethics)
