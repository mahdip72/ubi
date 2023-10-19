import torch


class CustomModel(torch.nn.Module):
    def __init__(self, model, hidden_size, mid_token):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.classifier = torch.nn.Linear(hidden_size, 2)
        self.model = model
        self.mid_token = mid_token

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        sequence_pred = self.model(x)[0]
        mid_pred = sequence_pred[:, self.mid_token, :]
        pred = self.classifier(mid_pred.squeeze())
        return pred


class LSTMModel(torch.nn.Module):
    def __init__(self, vocab_size, mid_token):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.mid_token = mid_token
        self.emb = torch.nn.Embedding(vocab_size,
                                  256,
                                  padding_idx=None,
                                  max_norm=None,
                                  device=None,
                                  dtype=None)
        self.lstm = torch.nn.LSTM(input_size=256, hidden_size=32, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = torch.nn.Linear(64, 2)

    def forward(self, sentence):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        embeds = self.emb(sentence)
        lstm_out, _ = self.lstm(embeds)
        pred = self.classifier(lstm_out[:, self.mid_token, :])
        return pred


def lstm(tokenizer, configs, device, pretrained_weights):
    if pretrained_weights:
        max_len = configs['window_size'] + 2
        vocab_size = len(tokenizer)
    else:
        max_len = configs['window_size']
        vocab_size = len(tokenizer) + 1

    model = LSTMModel(vocab_size=vocab_size, mid_token=int((configs['window_size'] + 1) / 2))

    # print('Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = model.to(device)
    return model


def prepare_model(device, configs, tokenizer, pretrained_weights=False, print_params=True):
    if configs['backbone'] == 'lstm':
        model = lstm(tokenizer, configs, device, pretrained_weights=False)
    else:
        print('wrong given backbone')
        model = None
        exit()
    if print_params:
        print('Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model
