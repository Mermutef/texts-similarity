from transformers import BertModel, BertTokenizer
import torch
import numpy as np


def bert(df, model_name):
    model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    doc = df[['answer']].values
    eta_answer = df[['true_answer']].values[0]
    embeddings = []
    for row in doc:
        encoded_input = tokenizer(row.answer, return_tensors='pt')
        with torch.no_grad():
            last_hidden_states = model(**encoded_input)[0]
        embedding = last_hidden_states.mean(1)
        embeddings.append(embedding)
    embeddings = [embedding.numpy()[0] for embedding in embeddings]
    encoded_input = tokenizer(eta_answer.answer, return_tensors='pt')
    with torch.no_grad():
        last_hidden_states = model(**encoded_input)[0]
    embedding = last_hidden_states.mean(1)
    print(f"{model_name} DONE")
    return {"answers": np.array(embeddings), "true_answer": embedding.numpy()[0]}
