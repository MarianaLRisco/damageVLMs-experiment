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
    train_path = os.path.join(root, "train_data.csv")
    test_path = os.path.join(root, "test_data.csv")
    val_path = os.path.join(root, "val_data.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    val_df = pd.read_csv(val_path)

    # Normalizar columnas
    for df in [train_df, test_df, val_df]:
        df.rename(columns={"label": "labels"}, inplace=True)
        # Paths in the clean CSVs are relative to root (e.g. "fires/images/foo.jpg")
        df["image_path"] = df["image_path"].apply(
            lambda p: os.path.join(root, p.replace("data/damage_dataset/", "")) if not os.path.isabs(p) else p
        )
        df["post_text"] = df["post_text"].fillna("")
        df["post_text"] = df["post_text"].apply(lambda x: str(x) if isinstance(x, str) else "") 

    # # Crear split de validación (10% del train)
    # train_df = train_df.reset_index(drop=True)
    # val_df = val_df.reset_index(drop=True)
    # test_df = test_df.reset_index(drop=True)

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
        df["post_text"] = df["post_text"].fillna("")
        df["post_text"] = df["post_text"].apply(lambda x: str(x) if isinstance(x, str) else "") 

    for df in [train_df, val_df, test_df]:
        df.reset_index(drop=True, inplace=True)

    return train_df, val_df, test_df
