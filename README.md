Applying Probe to LLM
=============
"[Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs](https://arxiv.org/abs/2306.13063)"의 result를 Phi-3로 재구성한 결과에 Azaria and Mitchell의 "[The Internal State of an LLM Knows When It's Lying.](https://arxiv.org/abs/2304.13734)"에서 소개된 Probe를 추가하여 성능 개선 관찰

## Overview
1) 이전에 작성한 result-regeneration 코드를 변형하여 feed-forward neural network를 Phi-3의 32개 layer 중 마지막 layer에 추가
+ result-regeneration 코드: [https://github.com/macroboomin/result-regeneration](https://github.com/macroboomin/result-regeneration)

2) Azaria and Mitchell의 "[The Internal State of an LLM Knows When It's Lying.](https://arxiv.org/abs/2304.13734)" 논문에서 제공한 데이터셋으로 neural network training 진행

3) training된 Probe model로 모델이 답안을 제공할 때 정답을 제공하였을 확률을 계산하여 vanilla prompt만으로 얻은 verbalized confidence와 비교

## File overview

### Verbalized Confidence
- `vanilla_prompt.py`: 모델에게 vanilla prompt를 주어 output을 csv 파일로 verbalized_results에 저장
- `verblalized_metrics.py`: verbalized_results에 있는 저장한 csv 파일로부터 ECE, AUROC, AUPRC-Positive, AUPRC-Negative 도출
- `verbalized_visual.py`: verbalized_results에 있는 저장한 csv 파일로부터 confidence-count, confidence-accuracy 그래프 도출

### Probe
- `probe.py`: probe 모델을 정의하고 data 폴더에 있는 데이터들로 neural network 학습 진행 후 'probe_mode.pth'로 저장
- `probe_prompt.py`: 'probe_mode.pth'을 load한 이후 모델에게 prompt를 주어 ouput을 얻고, 불러온 모델을 이용하여 output이 correct할 확률 계산하여 probe_results에 저장
- `probe_metrics`: probe_results에 있는 저장한 csv 파일로부터 ECE, AUROC, AUPRC-Positive, AUPRC-Negative 도출
- `probe_visual`: probe_results에 있는 저장한 csv 파일로부터 confidence-count, confidence-accuracy 그래프 도출

### utils
- `func.py`: 유틸로 사용 가능한 함수들을 저장 (e.g. extract_answer_and_confidence)
- `phi3.py`: Phi-3 모델을 불러온 이후 prompt를 연속적으로 생성

## Workflow

### Verbalized Confidence
1) `vanilla_prompt.py`를 이용하여 verbalized confidence 결과를 얻어서 csv 파일을 verbalized_results에 저장
2) `verbalized_metrics.py`를 이용하여 여러 metrics 도출
3) `verbalized_visual.py`를 이용하여 여러 그래프 도출

sh 폴더의 verbalized_confidence.sh로 한 번에 실행 가능

### Probe
0) `probe.py`로 Probe 모델을 만들고 데이터로 학습 후 모델을 torch를 이용하여 저장
1) `probe_prompt.py`를 이용하여 verbalized confidence 결과를 얻어서 csv 파일을 verbalized_results에 저장
2) `probe_metrics.py`를 이용하여 여러 metrics 도출
3) `probe_visual.py`를 이용하여 여러 그래프 도출

sh 폴더의 probe.sh로 한 번에 실행 가능

## Dataset

### Probe Training
+ Azaria and Mitchell의 "[The Internal State of an LLM Knows When It's Lying.](https://arxiv.org/abs/2304.13734)" 논문에서 직접 배포한 데이터셋([https://azariaa.com/Content/Datasets/true-false-dataset.zip.](https://azariaa.com/Content/Datasets/true-false-dataset.zip.))을 이용하여 Probe model training 하였음. Neural Network 아키텍쳐는 논문의 것과 동일하게 구성.

### Experiments
open-number나 open-ended 질문에 대하여서는 너무 낮은 수치의 accuracy가 나와서 모두 multiple choice 질문에 대하여서만 진행
+ Arithmetic reasoning: College Mathematics from MMLU (Col_Math)
+ Professional Knowledge: Professional-Law from MMLU (Prf_Law)
+ Ethical Knowledge: Business Ethics from MMLU (Biz_Ethics)
