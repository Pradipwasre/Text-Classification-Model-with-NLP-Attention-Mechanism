import torch


class TextDataset(torch.utils.data.Dataset):

    def __init__(self, tokens, embeddings, labels):
        """
        :param tokens: List of word tokens
        :param embeddings: Word embeddings (from glove)
        :param labels: List of labels
        """
        self.tokens = tokens
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        emb = torch.tensor(self.embeddings[self.tokens[idx], :])
        input_ = torch.cat((torch.ones(emb.shape[0],1), emb), dim=1)
        return torch.tensor(self.labels[idx]), input_
