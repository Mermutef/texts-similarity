from embeddings_creators.bert import bert
from embeddings_creators.gpt import gpt, train_gpt
from embeddings_creators.mamba import mamba
from metrics_calculating.distances_calculator import calc_all_distance_metrics
from metrics_calculating.quality_calculator import metrics
from result_presenters.visualizer import visualize

import pandas as pd

custom_df = pd.read_csv("similarity_phrases.csv", delimiter=';')
true_false_df = pd.read_csv("train (1).csv", delimiter=',', encoding="windows-1251")

# Bert
bert_models = [
    "bert-base-cased",
    "bert-base-uncased",
    "bert-large-cased",
    "bert-large-uncased",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking"
]

# GPT
gpt_models = [
    "openai-community/gpt2",
    "openai-community/gpt2-medium",
    "openai-community/gpt2-large"
]

train_gpt_models = [
    "sembeddings/model_gpt_trained",
    "sembeddings/gptops_finetuned_mpnet_gpu_v1"
]

# Mamba
mamba_models = [
    "Q-bert/Mamba-130M",
    "Q-bert/Mamba-1B"
]

# Stylometrics
stylos = [
    "all_stylometric_features",
    "char_pos_features",
    "char_word_features",
    "chars_features",
    "pos_features",
    "pos_word_features",
    "word_features"
]

distance_metrics = [
    "chebyshev",
    "cosine_similarity",
    "euclidean",
    "minkowski",
    "pearsonr"
]

# calculate for custom corpus
df = custom_df
data = df[["unique_id", "text1", "text2", "similarity"]]
doc = []
for _, i in data.iterrows():
    doc.append([i[['unique_id']].values[0], i[['text1']].values[0], i[['similarity']].values[0]])
    doc.append([i[['unique_id']].values[0], i[['text2']].values[0], i[['similarity']].values[0]])
df = pd.DataFrame(doc, columns=["group_id", "text", "similarity"])

print(df)

for gpt_model in gpt_models:
    gpt(df, gpt_model, "embeddings/gpt/custom_corpus")
    calc_all_distance_metrics("embeddings/gpt/custom_corpus", gpt_model)
    metrics(gpt_model,
            pd.read_csv(f"embeddings/gpt/true_false_corpus/{gpt_model.replace("/", "--")}-metrics.csv",
                        delimiter=","),
            "embeddings/gpt/true_false_corpus/results")

for bert_model in bert_models:
    bert(df, bert_model, "embeddings/bert/custom_corpus")
    calc_all_distance_metrics("embeddings/bert/custom_corpus", bert_model)
    metrics(bert_model,
            pd.read_csv(f"embeddings/bert/true_false_corpus/{bert_model.replace("/", "--")}-metrics.csv",
                        delimiter=","),
            "embeddings/bert/true_false_corpus/results")

#
for mamba_model in mamba_models:
    mamba(df, mamba_model, "embeddings/mamba/custom_corpus")
    calc_all_distance_metrics("embeddings/mamba/custom_corpus", mamba_model)
    metrics(mamba_model,
            pd.read_csv(f"embeddings/mamba/true_false_corpus/{mamba_model.replace("/", "--")}-metrics.csv",
                        delimiter=","),
            "embeddings/mamba/true_false_corpus/results")

for train_gpt_model in train_gpt_models:
    train_gpt(df, train_gpt_model, "embeddings/train_gpt/custom_corpus")
    calc_all_distance_metrics("embeddings/train_gpt/custom_corpus", train_gpt_model)
    metrics(train_gpt_model,
            pd.read_csv(f"embeddings/train_gpt/true_false_corpus/{train_gpt_model.replace("/", "--")}-metrics.csv",
                        delimiter=","),
            "embeddings/train_gpt/true_false_corpus/results")

for stylo_type in stylos:
    calc_all_distance_metrics("embeddings/stylo/custom_corpus", f"Custom_corpus_{stylo_type}")
    metrics(f"Custom_corpus_{stylo_type}",
            pd.read_csv(f"embeddings/stylo/custom_corpus/Custom_corpus_{stylo_type}-metrics.csv", delimiter=","),
            "embeddings/stylo/custom_corpus/results")
    for distance in distance_metrics:
        visualize(
            pd.read_csv(f"embeddings/stylo/custom_corpus/results/Custom_corpus_{stylo_type}-{distance}-results.csv",
                        delimiter=","),
            f"Custom_corpus_{stylo_type}_{distance}")

# calculate for train(1) corpus
df = true_false_df
data = df[["unique_id", "text1", "text2", "similarity"]]
doc = []
for _, i in data.iterrows():
    doc.append([i[['unique_id']].values[0], i[['text1']].values[0], i[['similarity']].values[0]])
    doc.append([i[['unique_id']].values[0], i[['text2']].values[0], i[['similarity']].values[0]])
df = pd.DataFrame(doc, columns=["group_id", "text", "similarity"])

print(df)

for gpt_model in gpt_models:
    gpt(df, gpt_model, "embeddings/gpt/true_false_corpus")
    calc_all_distance_metrics("embeddings/gpt/true_false_corpus", gpt_model)
    metrics(gpt_model,
            pd.read_csv(f"embeddings/gpt/true_false_corpus/{gpt_model.replace("/", "--")}-metrics.csv",
                        delimiter=","),
            "embeddings/gpt/true_false_corpus/results")

for bert_model in bert_models:
    bert(df, bert_model, "embeddings/bert/true_false_corpus")
    calc_all_distance_metrics("embeddings/bert/true_false_corpus", bert_model)
    metrics(bert_model,
            pd.read_csv(f"embeddings/bert/true_false_corpus/{bert_model.replace("/", "--")}-metrics.csv",
                        delimiter=","),
            "embeddings/bert/true_false_corpus/results")

for mamba_model in mamba_models:
    mamba(df, mamba_model, "embeddings/mamba/true_false_corpus")
    calc_all_distance_metrics("embeddings/mamba/true_false_corpus", mamba_model)
    metrics(mamba_model,
            pd.read_csv(f"embeddings/mamba/true_false_corpus/{mamba_model.replace("/", "--")}-metrics.csv",
                        delimiter=","),
            "embeddings/mamba/true_false_corpus/results")

for train_gpt_model in train_gpt_models:
    train_gpt(df, train_gpt_model, "embeddings/train_gpt/true_false_corpus")
    calc_all_distance_metrics("embeddings/train_gpt/true_false_corpus", train_gpt_model)
    metrics(train_gpt_model,
            pd.read_csv(f"embeddings/train_gpt/true_false_corpus/{train_gpt_model.replace("/", "--")}-metrics.csv",
                        delimiter=","),
            "embeddings/train_gpt/true_false_corpus/results")

for stylo_type in stylos:
    calc_all_distance_metrics("embeddings/stylo/true_false_corpus", f"TS_false_true_{stylo_type}")
    metrics(f"TS_false_true_{stylo_type}",
            pd.read_csv(f"embeddings/stylo/true_false_corpus/Custom_corpus_{stylo_type}-metrics.csv", delimiter=","),
            "embeddings/stylo/true_false_corpus/results")
    for distance in distance_metrics:
        visualize(
            pd.read_csv(f"embeddings/stylo/true_false_corpus/results/Custom_corpus_{stylo_type}-{distance}-results.csv",
                        delimiter=","),
            f"TS_false_true_{stylo_type}_{distance}")
