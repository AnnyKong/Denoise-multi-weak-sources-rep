import numpy as np
import pandas as pd
import torch
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

def build_dataset_from_raw():
    prefix = "Replace it with your own url"
    prefix2 = "./"
    df = pd.read_csv(prefix + "local_0120.csv")

    FB_news = df[['message']][:1000].copy()
    FB_news = FB_news.rename(columns={"message": "text"})
    FB_news.to_csv(prefix2 + "Fbnews.csv")

def choose_labeled_columns():
    prefix2 = "./"
    df = pd.read_csv(prefix2 + "fbnews_labeled.csv")
    df = df[:100]
    df.to_csv(prefix2 + "fbnews_labeled.csv")

def df_to_dict(df, nlp):
    target_dict = {}
    target_dict['text'] = df['text'].tolist()
    target_dict['label'] = torch.tensor(df['tag'].values)
    target_dict['major_label'] = torch.tensor(df['major_label'].values)
    target_dict['lf'] = torch.tensor(df[['LF1', 'LF2', 'LF3', 'LF4',
                                         'LF5', 'LF6', 'LF7', 'LF8',
                                         'LF9']].values)
    features = nlp(target_dict['text'])
    features = np.squeeze(features).astype('float32')[:, 0, :]
    target_dict['bert_feature'] = torch.tensor(features)
    return target_dict



if __name__ == '__main__':
    prefix2 = "./"
    df = pd.read_csv(prefix2 + "fbnews_LF.csv")
    df = df.sample(frac=1).reset_index(drop=True)
    # Add bert features
    text = df['text'].tolist()
    nlp = pipeline("feature-extraction",model='bert-base-uncased', tokenizer='bert-base-uncased')

    n, _ = df.shape
    labeled_size = int(0.1 * n)
    val_size = int(0.1 * n)
    test_size = int(0.1 * n)
    unlabeled_size = n - (labeled_size + val_size + test_size)

    labeled = df[:labeled_size]
    labeled_dict = df_to_dict(labeled, nlp)

    unlabeled = df[labeled_size:labeled_size + unlabeled_size]
    unlabeled_dict = df_to_dict(unlabeled, nlp)

    val = df[labeled_size + unlabeled_size : labeled_size + unlabeled_size + val_size]
    val_dict = df_to_dict(val, nlp)

    test = df[labeled_size + unlabeled_size + val_size:]
    test_dict = df_to_dict(test, nlp)




    data_dict = {"labeled": labeled_dict, "unlabeled": unlabeled_dict,
                 "validation": val_dict, "test": test_dict}

    torch.save(data_dict, prefix2 + "fbnews_organized_nb.pt")


