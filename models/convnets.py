import torch
import torch.nn as nn
import torchvision.models as models


def build_model(model_name: str, num_classes: int, trainable_layers: int = 2):

    # -----------------------
    # 1. Load model
    # -----------------------
    if model_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        feature_layers = list(model.children())[:-1]  # exclude fc

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        feature_layers = list(model.features)

    elif model_name == "vgg16":
        model = models.vgg16(weights="IMAGENET1K_V1")
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        feature_layers = list(model.features)



    else:
        raise ValueError(f"Model {model_name} not supported")

  
    for param in model.parameters():
        param.requires_grad = False


    for layer in feature_layers[-trainable_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

 
    for param in model.parameters():
        if param.requires_grad is False:
            continue

    if hasattr(model, "fc"):
        for param in model.fc.parameters():
            param.requires_grad = True

    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model