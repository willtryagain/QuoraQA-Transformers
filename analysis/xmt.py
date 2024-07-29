from datasets import load_dataset

dataset = load_dataset("toughdata/quora-question-answer-dataset")
question_set = {}
for i, q in enumerate(dataset["train"]["question"]):
    if q not in question_set:
        question_set[q] = []
    question_set[q].append(i)


from transformers import pipeline
import numpy as np
from math import  ceil
from tqdm import tqdm
import json

threshold = 0.20
default_max_length = 512

answer_judge = pipeline("zero-shot-classification", model= "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", device=0)

for i, q in tqdm(enumerate(question_set)):
    scores = [] 
    for index in question_set[q]:
        answer = dataset["train"][index]['answer']
        prompt = f"""
        Does the following answer reliably answer the question 
        {q}
        Provide a yes/no response. 
        Answer: {answer}
        """
        candidate_labels = ["yes"]
        res = answer_judge(prompt, candidate_labels)
        scores.append(res['scores'][0])
    sorted_indices = np.argsort(scores)
    total_selects = ceil(len(sorted_indices) * threshold)
    sorted_indices = sorted_indices[-total_selects:].tolist()
    question_set[q] = sorted_indices
    
    with open("../log/q_{}.json".format(i), "w") as f:
        json.dump({q : sorted_indices}, f)

