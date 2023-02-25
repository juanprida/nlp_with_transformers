# Natural Language Processing with Transformers. PyTorch Code Implementation.
This repository contains PyTorch code implementation for book Natural Language Processing with Transformers.
 
### Chapter 1: Hello Transformers
 - Introduction to Transformers - no code added.
### Chapter 2: Text classification.
    - We build a text classifier using distilbert-base-uncased model.
    - We follow two approaches: 
        - We take the last hidden state and we train a classifier on top of it.
        - We fine-tune the distilbert-base-uncased model.
### Chapter 3: Trasformer anatomy.
    - We build a the architecture for a transformer model. It can be used either as a decoder or encoder.
### Chapter 4: Multilingual Named Entity Recognition.
    - We build a multilingual named entity recognition model using the XLM-Roberta model.
    - We work with a dataset that contains 4 languages.
        - We only fine tune the model on the German language. In the book implementation they fine tune the model first on german and later on all the languages.