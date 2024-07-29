import pandas as pd
import re
from tqdm import tqdm

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

##### Vanilla prompts #####
from utils.phi3 import Phi3ChatCompletion
from utils.func import *
vanilla = {"role": "system", "content": "Read the question, provide your answer and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.\nUse the following format to answer:\n```Answer and Confidence (0-100): [ONLY the {answer_number}; not a complete sentence or symbols], [Your confidence level, please only include the numerical number in the range of 0-100]%```\nOnly the answer and confidence, don't give me the explanations."}

# Col_Math
messages = [
    vanilla,
    #{"role": "user", "content":f"Question: {Col_Math.loc[0]['question']}\nOptions: 1. {Col_Math.loc[0]['choices'][0]}\n2. {Col_Math.loc[0]['choices'][1]}\n3. {Col_Math.loc[0]['choices'][2]}\n4. {Col_Math.loc[0]['choices'][3]}"},
    #{"role": "assistant", "content": "Answer and Confidence (0-100): 1, 90%"},
    {"":""},
    {"":""}, # dummy
]

for index, row in tqdm(Col_Math.iterrows(), total=Col_Math.shape[0], desc="Processing Col_Math rows"):
    if index == 0: continue
    messages.pop()
    messages.pop()
    messages.append(vanilla)
    messages.append({"role": "user", "content":f"Question: {Col_Math.loc[index]['question']}\nOptions: 1. {Col_Math.loc[index]['choices'][0]}\n2. {Col_Math.loc[index]['choices'][1]}\n3. {Col_Math.loc[0]['choices'][2]}\n4. {Col_Math.loc[index]['choices'][3]}"})
    
    gen_answer, confidence = extract_answer_and_confidence(Phi3ChatCompletion(messages))
    if gen_answer == 0 and confidence == 0: continue
    
    Col_Math.at[index, 'gen_answer'] = gen_answer
    Col_Math.at[index, 'confidence'] = confidence
    Col_Math.at[index, 'correct'] = 1 if gen_answer == row['answer'] else 0

Col_Math.dropna(subset=['gen_answer'], inplace=True)
print(Col_Math)

Col_Math.to_csv('./verbalized_results/Col_Math_processed.csv', index=False)
print("Col_Math dataset saved to Col_Math_processed.csv")


# Biz_Ethics
messages = [
    vanilla,
    #{"role": "user", "content":f"Question: {Biz_Ethics.loc[0]['question']}\nOptions: 1. {Biz_Ethics.loc[0]['choices'][0]}\n2. {Biz_Ethics.loc[0]['choices'][1]}\n3. {Biz_Ethics.loc[0]['choices'][2]}\n4. {Biz_Ethics.loc[0]['choices'][3]}"},
    #{"role": "assistant", "content": "Answer and Confidence (0-100): 1, 90%"},
    {"":""},
    {"":""}, # dummy
]

for index, row in tqdm(Biz_Ethics.iterrows(), total=Biz_Ethics.shape[0], desc="Processing Biz_Ethics rows"):
    if index == 0: continue
    messages.pop()
    messages.pop()
    messages.append(vanilla)
    messages.append({"role": "user", "content":f"Question: {Biz_Ethics.loc[index]['question']}\nOptions: 1. {Biz_Ethics.loc[index]['choices'][0]}\n2. {Biz_Ethics.loc[index]['choices'][1]}\n3. {Biz_Ethics.loc[0]['choices'][2]}\n4. {Biz_Ethics.loc[index]['choices'][3]}"})
    
    gen_answer, confidence = extract_answer_and_confidence(Phi3ChatCompletion(messages))
    if gen_answer == 0 and confidence == 0: continue
    
    Biz_Ethics.at[index, 'gen_answer'] = gen_answer
    Biz_Ethics.at[index, 'confidence'] = confidence
    Biz_Ethics.at[index, 'correct'] = 1 if gen_answer == row['answer'] else 0

Biz_Ethics.dropna(subset=['gen_answer'], inplace=True)
print(Biz_Ethics)

Biz_Ethics.to_csv('./verbalized_results/Biz_Ethics_processed.csv', index=False)
print("Biz_Ethics dataset saved to Biz_Ethics_processed.csv")


# Prf_Law
messages = [
    vanilla,
    #{"role": "user", "content":f"Question: {Prf_Law.loc[0]['question']}\nOptions: 1. {Prf_Law.loc[0]['choices'][0]}\n2. {Prf_Law.loc[0]['choices'][1]}\n3. {Prf_Law.loc[0]['choices'][2]}\n4. {Prf_Law.loc[0]['choices'][3]}"},
    #{"role": "assistant", "content": "Answer and Confidence (0-100): 3, 95%"},
    {"":""},
    {"":""}, # dummy
]

for index, row in tqdm(Prf_Law.iterrows(), total=Prf_Law.shape[0], desc="Processing Prf_Law rows"):
    if index == 0: continue
    messages.pop()
    messages.pop()
    messages.append(vanilla)
    messages.append({"role": "user", "content":f"Question: {Prf_Law.loc[index]['question']}\nOptions: 1. {Prf_Law.loc[index]['choices'][0]}\n2. {Prf_Law.loc[index]['choices'][1]}\n3. {Prf_Law.loc[index]['choices'][2]}\n4. {Prf_Law.loc[index]['choices'][3]}"})
    
    gen_answer, confidence = extract_answer_and_confidence(Phi3ChatCompletion(messages))
    if gen_answer == 0 and confidence == 0: continue
    
    Prf_Law.at[index, 'gen_answer'] = gen_answer
    Prf_Law.at[index, 'confidence'] = confidence
    Prf_Law.at[index, 'correct'] = 1 if gen_answer == row['answer'] else 0

Prf_Law.dropna(subset=['gen_answer'], inplace=True)
print(Prf_Law)

Prf_Law.to_csv('./verbalized_results/Prf_Law_processed.csv', index=False)
print("Prf_Law dataset saved to Prf_Law_processed.csv")
