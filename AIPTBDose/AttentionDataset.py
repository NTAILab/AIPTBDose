from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
import numpy as np
from typing import Dict

def survival_pairing(df: pd.DataFrame, input_dim: int, equal_inputs: bool = False):
    df_t = df[df["a"] != 0].reset_index(drop=True)
    df_t = df_t.rename(columns={'a': 'az', 'y': 'h', 'c': 'ct'})
    for i in range(1, input_dim+1):
        df_t = df_t.rename(columns={f'x{i}': f'z{i}'})

    df_c = df[df["a"] == 0].reset_index(drop=True)
    df_c = df_c.rename(columns={'a': 'ax', 'c': 'cc'})

    if equal_inputs:
        df_pairs = df_c.merge(df_t, left_index=True, right_index=True)
    else:
        df_pairs = df_c.merge(df_t, how='cross')
    df_pairs["te"] = df_pairs["h"] - df_pairs["y"]
    df = df_pairs[~((df_pairs['cc'] == 0) & (df_pairs['ct'] == 0))]
    return df


class AttentionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, scaler: StandardScaler, feature_cols: list[str]):
        if 'az' in feature_cols:
            feature_cols.remove('az')
            feature_cols.insert(0, 'az')
        self.features = torch.tensor(scaler.transform(df[feature_cols]), dtype=torch.float32)
        self.values = torch.tensor((df['te'] > 0).astype(np.float32).values, dtype=torch.float32).unsqueeze(1) 

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if idx == "feature":
            return self.features
        if idx == "value":
            return self.values

        item = {
            'feature': self.features[idx],
            'value': self.values[idx],
        }

        return item


def create_attention_datasets(df: pd.DataFrame,
                    train_idx: list[int],
                    val_idx: list[int],
                    dim: int) -> tuple[Dict[str, AttentionDataset], StandardScaler]:

    features = ["az"]
    for i in range(1, dim + 1):
        features += [f'x{i}', f'z{i}']

    train_df = survival_pairing(df.iloc[train_idx], dim, False)
    val_df = survival_pairing(df.iloc[val_idx], dim, False)

    scaler = StandardScaler()
    scaler.fit(train_df[features])

    train_ds = AttentionDataset(train_df, scaler, features)
    val_ds = AttentionDataset(val_df, scaler, features)
    datasets = {
        "train": train_ds,
        "val1": val_ds,
    }

    # additional validation dataset for classifying IPTB
    if "y_test" in df.columns:
        df_test = df.iloc[val_idx].copy()

        y_values = np.where(df_test['a'] == 0, df_test['y_test'], df_test['y'])
        y_test_values = np.where(df_test['a'] == 0, df_test['y'], df_test['y_test'])
        a_values = np.where(df_test['a'] == 0, df_test['a_test'], df_test['a'])
        a_test_values = np.where(df_test['a'] == 0, df_test['a'], df_test['a_test'])
        df_test['y'] = y_values
        df_test['y_test'] = y_test_values
        df_test['a'] = a_values
        df_test['a_test'] = a_test_values

        features_train = ["a", "y", "c"]
        for i in range(1, dim + 1):
            features_train += [f'x{i}']
        features_test = ["a_test", "y_test", "c_test"]
        for i in range(1, dim + 1):
            features_test += [f'x{i}_test']
        features_control = ["ax", "y", "cc"]
        for i in range(1, dim + 1):
            features_control += [f'x{i}']
        features_treat = ["az", "h", "ct"]
        for i in range(1, dim + 1):
            features_treat += [f'z{i}']
        df_test = df_test.rename(columns=dict(zip(features_train, features_treat)))
        df_test = df_test.rename(columns=dict(zip(features_test, features_control)))
        df_test['te'] = df_test['h'] - df_test['y']

        test_ds = AttentionDataset(df_test, scaler, features)
        datasets["val2"] = test_ds

    return datasets, scaler