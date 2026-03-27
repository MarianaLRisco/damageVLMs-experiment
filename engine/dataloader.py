from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataloaders.sigmoidCrossentropy import ImageTextDataset
from dataloaders.twoStage import ImageTextTwoStage
from dataloaders.sentenceTransformers import ImageTextSentenceTransformers, custom_collate_fn
from dataloaders.mclip import ImageTextMclip


DATASET_MAP = {
    "sigmoid": ImageTextDataset,
    "crossentropy": ImageTextDataset,
    "twoStage": ImageTextTwoStage,
    "sentenceTransformers": ImageTextSentenceTransformers,
    "mclip": ImageTextMclip,
}



def get_dataloader(model_type, data, processor=None, tokenizer=None, preprocess=None,
                   batch_size=32):
    # Split data
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_df, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

    if model_type not in DATASET_MAP and model_type != "clip":
        raise ValueError(f"Modelo no soportado: {model_type}")

    if model_type == "clip":
        return train_df, val_data, test_data  

    DatasetClass = DATASET_MAP[model_type]

    if model_type == "mclip":
        dataset_kwargs = {"tokenizer": tokenizer, "preprocess": preprocess}
    else:
        dataset_kwargs = {"processor": processor}

    train_dataset = DatasetClass(data=train_df, **dataset_kwargs)
    val_dataset   = DatasetClass(data=val_data, **dataset_kwargs)
    test_dataset  = DatasetClass(data=test_data, **dataset_kwargs)


    loader_args = dict(batch_size=batch_size, num_workers=1)

    # =========================================================
    # 5. COLLATE only for Sentence Transformers
    # =========================================================
    collate = custom_collate_fn if model_type == "sentenceTransformers" else None

    train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=collate, **loader_args)
    val_loader   = DataLoader(val_dataset, shuffle=False, collate_fn=collate, **loader_args)
    test_loader  = DataLoader(test_dataset, shuffle=False, num_workers=2, collate_fn=collate, batch_size=batch_size)

    return train_loader, val_loader, test_loader
