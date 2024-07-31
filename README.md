# QuoraQA-Transformers

 Analysing the performance of transformers on Quora QA dataset,

Here we explored the Quora QA dataset, cleaned and found interesting topics. We fine-tuned multiple models on the reduced dataset and found that BART performs better that T5. This can be attributed to the fact that T5 has more parameters and requires more data. We also perfomed data augmentation but observed that it can lead to overfitting if it is too similar to exisiting training data specially when using high learning rate.We perfomed an interesting experiment involving both answer and question prediction which results in singificant loss in test loss.

libs: nlpaug, matplotlib, numpy, huggingface, pytorch, gensim, nltk, spacy

open analysis/quora-qa.ipynb
