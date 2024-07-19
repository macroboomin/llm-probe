Result Regeneration
=============
Partial result regeneration of "Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs"

Paper: [https://openreview.net/forum?id=gjeQKFxFpZ](https://openreview.net/forum?id=gjeQKFxFpZ)
-------------
# Oveview
2080 클러스터의 1GPU로 실험하기 적절하면서도 prompt를 잘 이해한다고 판단하여 마이크로소프트 사의 small language model인 Phi-3-mini-4k-instruct을 이용
model: [text](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

verbalized_confidence.sh 파일 실행으로 전체 코드를 한 번에 실행하는 것이 가능
'''
sh verbalized_confidence.sh     # In terminal
'''

# File overview

## main
### `vanilla_prompt.py`: 모델에게 vanilla prompt를 주어 output을 csv 파일로 저장
### `metrics.py`: 저장한 csv 파일로부터 ECE, AUROC, AUPRC-Positive, AUPRC-Negative 도출
### `visual.py`: 저장한 csv 파일로부터 confidence-count, confidence-accuracy 그래프 도출

## utils
### `func.py`: 유틸로 사용 가능한 함수들을 저장 (e.g. extract_answer_and_confidence)
### `phi3.py`: Phi-3 모델을 불러온 이후 prompt를 연속적으로 생성

# Dataset
논문의 정확한 데이터셋을 모두 구하지는 못하였고, 찾을 수 있었던 데이터셋들로만 진행

### Arithmetic reasoning: GSM8K (GSM8K)
### Professional Knowledge: Professional-Law from MMLU (Prf_Law)
### Ethical Knowledge: Business Ethics from MMLU (Biz_Ethics)
