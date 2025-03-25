from transformers import GPT2Tokenizer, GPT2Model
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd

from metrics_calculating.distances_calculator import calc_all_distance_metrics


def gpt(df, model_name, result_path):
    model = GPT2Model.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    doc = df[['text']].values
    embeddings = []
    for row in doc:
        encoded_input = tokenizer(row[0], return_tensors='pt')
        with torch.no_grad():
            last_hidden_states = model(**encoded_input)[0]
        embedding = last_hidden_states.mean(1)
        embeddings.append(embedding)
    embeddings = [i.numpy()[0] for i in embeddings]
    df1 = pd.DataFrame(embeddings)
    pd.concat([df, df1], axis=1).to_csv(f'{result_path}/{model_name.replace("/", "--")}.csv', index=False)
    calc_all_distance_metrics(result_path, model_name)
    print(f"{model_name} DONE")
    return


def train_gpt(df, model_name, result_path):
    model = SentenceTransformer(model_name)
    doc = df[['text']].values
    embeddings = []
    for row in doc:
        embedding = model.encode(row[0])
        embeddings.append(embedding)
    df1 = pd.DataFrame(embeddings)
    pd.concat([df, df1], axis=1).to_csv(f'{result_path}/{model_name.replace("/", "--")}.csv', index=False)
    calc_all_distance_metrics(result_path, model_name)
    print(f"{model_name} DONE")
    return
