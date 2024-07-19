import pandas as pd
import re
from tqdm import tqdm

##### Load datasets #####
splits = {'train': 'main/train-00000-of-00001.parquet', 'test': 'main/test-00000-of-00001.parquet'}
GSM8K = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["test"])

splits = {'test': 'business_ethics/test-00000-of-00001.parquet', 'validation': 'business_ethics/validation-00000-of-00001.parquet', 'dev': 'business_ethics/dev-00000-of-00001.parquet'}
Biz_Ethics = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])

splits = {'test': 'professional_law/test-00000-of-00001.parquet', 'validation': 'professional_law/validation-00000-of-00001.parquet', 'dev': 'professional_law/dev-00000-of-00001.parquet'}
Prf_Law = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])

##### Preprocessing #####
GSM8K['answer'] = GSM8K['answer'].str.extract(r'#### (\d+)')
GSM8K = GSM8K.dropna(subset=['answer'])
GSM8K['answer'] = GSM8K['answer'].astype(int)
GSM8K['gen_answer'] = None
GSM8K['confidence'] = None
GSM8K['correct'] = None

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

##### Vanilla prompts #####
from utils.phi3 import Phi3ChatCompletion
from utils.func import *
vanilla = {"role": "system", "content": "Read the question, provide your answer and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.\nUse the following format to answer:\n```Answer and Confidence (0-100): [ONLY the {answer_number}; not a complete sentence or symbols], [Your confidence level, please only include the numerical number in the range of 0-100]%```\nOnly the answer and confidence, don't give me the explanations."}

# Biz_Ethics
messages = [
    vanilla,
    {"role": "user", "content":f"Question: {Biz_Ethics.loc[0]['question']}\nOptions: 1. {Biz_Ethics.loc[0]['choices'][0]}\n2. {Biz_Ethics.loc[0]['choices'][1]}\n3. {Biz_Ethics.loc[0]['choices'][2]}\n4. {Biz_Ethics.loc[0]['choices'][3]}"},
    {"role": "assistant", "content": "Answer and Confidence (0-100): 1, 90%"},
    {"":""},
    {"":""}, # dummy
]

for index, row in tqdm(Biz_Ethics.iterrows(), total=Biz_Ethics.shape[0], desc="Processing Biz_Ethics rows"):
    if index == 0: continue
    messages.pop()
    messages.pop()
    messages.append(vanilla)
    messages.append({"role": "user", "content":f"Question: {row['question']}"})
    
    gen_answer, confidence = extract_answer_and_confidence(Phi3ChatCompletion(messages))
    if gen_answer == 0 and confidence == 0: continue
    
    Biz_Ethics.at[index, 'gen_answer'] = gen_answer
    Biz_Ethics.at[index, 'confidence'] = confidence
    Biz_Ethics.at[index, 'correct'] = 1 if gen_answer == row['answer'] else 0

Biz_Ethics.dropna(subset=['gen_answer'], inplace=True)
print(Biz_Ethics)

Biz_Ethics.to_csv('./data/Biz_Ethics_processed.csv', index=False)
print("Biz_Ethics dataset saved to Biz_Ethics_processed.csv")


# GSM8K
messages = [
    vanilla,
    {"role": "user", "content":f"Question: {GSM8K.loc[0]['question']}"},
    {"role": "assistant", "content": "Answer and Confidence (0-100): 72, 95%"},
    {"":""},
    {"":""}, # dummy
]

for index, row in tqdm(GSM8K.iterrows(), total=GSM8K.shape[0], desc="Processing GSM8K rows"):
    if index == 0: continue
    messages.pop()
    messages.pop()
    messages.append(vanilla)
    messages.append({"role": "user", "content":f"Question: {row['question']}"})
    
    gen_answer, confidence = extract_answer_and_confidence(Phi3ChatCompletion(messages))
    if gen_answer == 0 and confidence == 0: continue
    
    GSM8K.at[index, 'gen_answer'] = gen_answer
    GSM8K.at[index, 'confidence'] = confidence
    GSM8K.at[index, 'correct'] = 1 if gen_answer == row['answer'] else 0

GSM8K = GSM8K.dropna()
print(GSM8K)

GSM8K.to_csv('./data/GSM8K_processed.csv', index=False)
print("GSM8K dataset saved to GSM8K_processed.csv")


# Prf_Law
messages = [
    vanilla,
    {"role": "user", "content":f"Question: {Prf_Law.loc[0]['question']}\nOptions: 1. {Prf_Law.loc[0]['choices'][0]}\n2. {Prf_Law.loc[0]['choices'][1]}\n3. {Prf_Law.loc[0]['choices'][2]}\n4. {Prf_Law.loc[0]['choices'][3]}"},
    {"role": "assistant", "content": "Answer and Confidence (0-100): 3, 95%"},
    {"":""},
    {"":""}, # dummy
]

for index, row in tqdm(Prf_Law.iterrows(), total=Prf_Law.shape[0], desc="Processing Prf_Law rows"):
    if index == 0: continue
    messages.pop()
    messages.pop()
    messages.append(vanilla)
    messages.append({"role": "user", "content":f"Question: {row['question']}"})
    
    gen_answer, confidence = extract_answer_and_confidence(Phi3ChatCompletion(messages))
    if gen_answer == 0 and confidence == 0: continue
    
    Prf_Law.at[index, 'gen_answer'] = gen_answer
    Prf_Law.at[index, 'confidence'] = confidence
    Prf_Law.at[index, 'correct'] = 1 if gen_answer == row['answer'] else 0

Prf_Law.dropna(subset=['gen_answer'], inplace=True)
print(Prf_Law)

Prf_Law.to_csv('./data/Prf_Law_processed.csv', index=False)
print("Prf_Law dataset saved to Prf_Law_processed.csv")