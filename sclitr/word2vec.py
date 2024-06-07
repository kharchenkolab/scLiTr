import torch.nn as nn

class SkipGram(nn.Module):
    def __init__(self, z_dim, vocab_size, device):
        super(SkipGram, self).__init__()
        self.device = device

        self.embedding = nn.Embedding(
            vocab_size,
            z_dim,
        )
        self.output = nn.Linear(
            z_dim,
            vocab_size,
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_word):
        emb_input = self.embedding(input_word)
        context_scores = self.output(emb_input)
        log_ps = self.log_softmax(context_scores)
        
        return log_ps