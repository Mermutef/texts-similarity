import random

import pandas as pd

import warnings

from determining.random_settings import determine_random

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

determine_random()

def flatten(iterable):
    return [item for row in iterable for item in row]

df_map_1 = {}
df_map_2 = {}
df_map_3 = {}

excel_reader = pd.ExcelFile('1.xlsx')

for sheet_name in excel_reader.sheet_names:
    df_map_1[sheet_name] = excel_reader.parse(sheet_name, index_col=0)

excel_reader = pd.ExcelFile('2.xlsx')

for sheet_name in excel_reader.sheet_names:
    df_map_2[sheet_name] = excel_reader.parse(sheet_name, index_col=0)

excel_reader = pd.ExcelFile('3.xlsx')
for sheet_name in excel_reader.sheet_names:
    df_map_3[sheet_name] = excel_reader.parse(sheet_name, index_col=0)

answers = {}
with open("Ответы.txt", "r") as f:
    eta_answers = list(filter(lambda line: len(str(line).strip()) != 0, f.readlines()))
    eta_answers = list(map(lambda line: str(line).strip(), eta_answers))

answers_map = {}
for question_page in excel_reader.sheet_names:
    answers[question_page] = []
keyset = list(map(lambda x: x, answers.keys()))
for i in range(len(eta_answers)):
    answers_map[keyset[i]] = eta_answers[i]

for question_page in excel_reader.sheet_names:
    answers_1 = df_map_1[question_page]
    answers_2 = df_map_2[question_page]
    answers_3 = df_map_3[question_page]
    for i, _ in answers_1.iterrows():
        row_idx = i - 1
        mark_1 = answers_1.iloc[row_idx].mark
        mark_2 = answers_2.iloc[row_idx].mark
        mark_3 = answers_3.iloc[row_idx].mark

        marks = [
            1 if mark_1 == 1 or mark_1 == 0 else 0,
            1 if mark_2 == 1 or mark_2 == 0 else 0,
            1 if mark_3 == 1 or mark_3 == 0 else 0
        ]

        answer = str(answers_1.iloc[row_idx].answer).strip()
        mark = 1 if sum(marks) > 1.5 else 0
        true_answer = answers_map[question_page]

        answers[question_page].append({"answer": answer, "true_answer": true_answer, "mark": mark})

new_answers = {}
for question_page in excel_reader.sheet_names:
    new_answers[question_page] = []
    i = -1
    for page in excel_reader.sheet_names:
        if page != question_page:
            mark = 0
            rand_answers = random.sample(answers[page], 4 + i)
            i *= -1
            true_answer = answers_map[question_page]
            for answer in rand_answers:
                new_answers[question_page].append({"answer": answer, "true_answer": true_answer, "mark": mark})

df = pd.DataFrame(flatten(answers.values()))
df1 = pd.DataFrame(flatten(new_answers.values()))
print(sum(flatten(df[['mark']].values)))
print(len(flatten(df[['mark']].values)))
print(sum(flatten(df[['mark']].values)) * 1.0 / len(flatten(df[['mark']].values)))
print()
print(sum(flatten(df1[['mark']].values)))
print(len(flatten(df1[['mark']].values)))
print(sum(flatten(df1[['mark']].values)) * 1.0 / len(flatten(df1[['mark']].values)))
df2 = pd.concat([df, df1], axis=0)
print()
print(sum(flatten(df2[['mark']].values)))
print(len(flatten(df2[['mark']].values)))
print(sum(flatten(df2[['mark']].values)) * 1.0 / len(flatten(df2[['mark']].values)))

df2.to_csv('extended-corpus.csv', index=False)