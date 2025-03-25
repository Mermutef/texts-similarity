from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd

from metrics_calculating.distances_calculator import calc_all_distance_metrics


def mamba(df, model_name, result_path):
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    doc = df[['text']].values
    embeddings = []
    for row in doc:
        input_ids = tokenizer.encode(row[0], return_tensors="pt")
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0]
        embedding = last_hidden_states.mean(1)
        embeddings.append(embedding)
    embeddings = [i.numpy()[0] for i in embeddings]
    df1 = pd.DataFrame(embeddings)
    pd.concat([df, df1], axis=1).to_csv(f'{result_path}/{model_name.replace("/", "--")}.csv', index=False)
    calc_all_distance_metrics(result_path, model_name)
    print(f"{model_name} DONE")
