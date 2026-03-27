import torch
import torch.nn as nn

class FineTuneContrastive(nn.Module):
    def __init__(self, modelA, modelB, device='cuda'):
        super().__init__()
        self.modelA = modelA  
        self.modelB = modelB  
        self.device = device

    def embed_A(self, inputA):  #  PIL Images or paths
        tokenized = self.modelA.tokenize(inputA)
        tokenized = {key: torch.tensor(val).to(self.device) for key, val in tokenized.items()}


        out = self.modelA(tokenized)
        return out['sentence_embedding']  # Tensor gradient

    def embed_B(self, inputB):  # List of texts
        tokenized = self.modelB.tokenize(inputB)
        for key in tokenized:
            tokenized[key] = tokenized[key].to(self.device)

        out = self.modelB(tokenized)

        return out['sentence_embedding']  # Tensor gradient

    def forward(self, inputA, inputB):
        embedA = self.embed_A(inputA)
        embedB = self.embed_B(inputB)
        
        return embedA, embedB