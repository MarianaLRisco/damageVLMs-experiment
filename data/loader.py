import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Any

def load_damage_dataset(root: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga el damage_dataset desde sus CSVs pre-divididos.

    Args:
        root: ruta al directorio del dataset (e.g. 'dataset/damage_dataset')

    Returns:
        (train_df, val_df, test_df) con columnas: image_path, post_text, labels
    """
    train_path = os.path.join(root, "train_data_clean.csv")
    test_path = os.path.join(root, "test_data_clean.csv")

    train_full = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Normalizar columnas
    for df in [train_full, test_df]:
        df.rename(columns={"label": "labels"}, inplace=True)
        # Paths in the clean CSVs are relative to root (e.g. "fires/images/foo.jpg")
        df["image_path"] = df["image_path"].apply(
            lambda p: os.path.join(root, p) if not os.path.isabs(p) else p
        )

    # Crear split de validación (10% del train)
    train_result: Any = train_test_split(train_full, test_size=0.1, random_state=42, stratify=train_full["labels"])
    train_df_split, val_df_split = train_result
    train_df = train_df_split.reset_index(drop=True)
    val_df = val_df_split.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def load_crisisMMD(root: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga el CrisisMMD desde sus CSVs pre-divididos (train/dev/test).

    Args:
        root: ruta al directorio del dataset (e.g. 'dataset/crisisMMD')

    Returns:
        (train_df, val_df, test_df) con columnas: image_path, post_text, labels
    """
    train_df = pd.read_csv(os.path.join(root, "csv_splits", "train.csv"))
    val_df = pd.read_csv(os.path.join(root, "csv_splits", "dev.csv"))
    test_df = pd.read_csv(os.path.join(root, "csv_splits", "test.csv"))

    for df in [train_df, val_df, test_df]:
        df.rename(columns={
            "tweet_text": "post_text",
            "label": "labels",
            "image": "image_path",
        }, inplace=True)
        # Normalizar image_path: root + '/' + path relativo
        df["image_path"] = df["image_path"].apply(
            lambda p: os.path.join(root, p) if not os.path.isabs(p) else p
        )

    for df in [train_df, val_df, test_df]:
        df.reset_index(drop=True, inplace=True)

    return train_df, val_df, test_df
