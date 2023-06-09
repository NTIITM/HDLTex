import torch

"""
分布式表示
后续加入bert
"""


class embed_model(torch.nn.Module):
    def __init__(self, is_bert=False, num_embeddings=50000, embedding_dim=512, device=None):
        super().__init__()
        if is_bert:
            pass
        else:
            self.embeding = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
            # print(self.embeding.de)

    def forward(self, X):
        return self.embeding(X)
