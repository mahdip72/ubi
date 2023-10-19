import torch
# from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.tab_network import TabNet
from transformers import BertModel, BertConfig
from transformers import MobileBertModel, MobileBertConfig
from transformers import AlbertForSequenceClassification, AlbertConfig
from transformers import AlbertModel
from transformers import SqueezeBertModel, SqueezeBertConfig
from transformers import NystromformerModel, NystromformerConfig


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


class TabNetModel(torch.nn.Module):
    def __init__(self, vocab_size, window_size):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.window_size = window_size
        self.tabnet = TabNet(input_dim=window_size, output_dim=2,
                             n_d=64, n_a=64, n_steps=5,
                             cat_emb_dim=vocab_size,
                             cat_dims=[256]*self.window_size,
                             cat_idxs=list(range(0, self.window_size)),
                             virtual_batch_size=512)
        # self.classifier = torch.nn.Linear(128, 2)

    def forward(self, sentence):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # embeds = self.emb(sentence)
        tabnet_out, _ = self.tabnet(sentence)
        # pred = self.classifier(lstm_out[:, self.mid_token, :])
        return tabnet_out


def albert_classification(tokenizer, configs, device):
    model_config = AlbertConfig(
        vocab_size=len(tokenizer) + 1,
        max_position_embeddings=configs['window_size'],
        embedding_size=1024,
        # hidden_size=1024,
        # num_hidden_layers=4,
        num_labels=2,
        return_dict=False,
        # classifier_activation=True
    )
    # model = MobileBertForSequenceClassification(config)
    model = AlbertForSequenceClassification(model_config)
    # model = BertForSequenceClassification.from_pretrained("Rostlab/prot_bert", return_dict=False)
    # model = AlbertForSequenceClassification.from_pretrained("Rostlab/prot_albert", return_dict=False)
    # model = model.from_pretrained("Rostlab/prot_bert", return_dict=False)
    # model = BertModel.from_pretrained("Rostlab/prot_bert")

    # for parameter in model.parameters():
    #     print(parameter.shape)
    # print('Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = model.to(device)
    return model


def albert(tokenizer, configs, device, pretrained_weights):
    if pretrained_weights:
        max_len = configs['window_size'] + 2
        vocab_size = len(tokenizer)
    else:
        max_len = configs['window_size']
        vocab_size = len(tokenizer) + 1

    model_config = AlbertConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max_len,
        # embedding_size=512,
        # hidden_size=256,
        # num_attention_heads=8,
        # intermediate_size=1024,
        # num_hidden_layers=4,
        return_dict=False,
        # classifier_activation=True
    )
    model = AlbertModel(model_config)

    # print('Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if pretrained_weights:
        model = model.from_pretrained("Rostlab/prot_albert", return_dict=False)
        # print('no pretrained weights is considered')
        custom_model = CustomModel(model,
                                   hidden_size=model_config.hidden_size,
                                   mid_token=int((configs['window_size'] + 1) / 2) + 1)
    else:
        custom_model = CustomModel(model,
                                   hidden_size=model_config.hidden_size,
                                   mid_token=int((configs['window_size'] + 1) / 2))

    # for parameter in custom_model.parameters():
    #     print(parameter.shape)

    # print('Number of parameters:', sum(p.numel() for p in custom_model.parameters() if p.requires_grad))

    custom_model = custom_model.to(device)
    return custom_model


def mobile_bert(tokenizer, configs, device, pretrained_weights):
    if pretrained_weights:
        max_len = configs['window_size'] + 2
        vocab_size = len(tokenizer)
    else:
        max_len = configs['window_size']
        vocab_size = len(tokenizer) + 1

    model_config = MobileBertConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max_len,
        # embedding_size=768,
        # hidden_size=768,
        # num_attention_heads=8,
        # intermediate_size=1024,
        # num_hidden_layers=4,
        return_dict=False,
        # classifier_activation=True
    )
    model = MobileBertModel(model_config)

    # print('Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if pretrained_weights:
        print('no pretrained weights is considered')
        custom_model = None
    else:
        custom_model = CustomModel(model,
                                   hidden_size=model_config.hidden_size,
                                   mid_token=int((configs['window_size'] + 1) / 2))

    # for parameter in custom_model.parameters():
    #     print(parameter.shape)

    # print('Number of parameters:', sum(p.numel() for p in custom_model.parameters() if p.requires_grad))

    custom_model = custom_model.to(device)
    return custom_model


def squeeze_bert(tokenizer, configs, device, pretrained_weights):
    if pretrained_weights:
        max_len = configs['window_size'] + 2
        vocab_size = len(tokenizer)
    else:
        max_len = configs['window_size']
        vocab_size = len(tokenizer) + 1

    model_config = SqueezeBertConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max_len,
        # embedding_size=768,
        # hidden_size=768,
        # num_attention_heads=8,
        # intermediate_size=1024,
        # num_hidden_layers=4,
        return_dict=False,
        # classifier_activation=True
    )
    model = SqueezeBertModel(model_config)

    # print('Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if pretrained_weights:
        print('no pretrained weights is considered')
        custom_model = None
    else:
        custom_model = CustomModel(model,
                                   hidden_size=model_config.hidden_size,
                                   mid_token=int((configs['window_size'] + 1) / 2))
    # for parameter in custom_model.parameters():
    #     print(parameter.shape)

    # print('Number of parameters:', sum(p.numel() for p in custom_model.parameters() if p.requires_grad))

    custom_model = custom_model.to(device)
    return custom_model


def nystromformer(tokenizer, configs, device, pretrained_weights):
    if pretrained_weights:
        max_len = configs['window_size'] + 2
        vocab_size = len(tokenizer)
    else:
        max_len = configs['window_size']
        vocab_size = len(tokenizer) + 1

    model_config = NystromformerConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max_len,
        # embedding_size=768,
        # hidden_size=768,
        # num_attention_heads=6,
        # intermediate_size=1024,
        num_hidden_layers=6,
        return_dict=False,
        # classifier_activation=True
    )
    model = NystromformerModel(model_config)

    # print('Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if pretrained_weights:
        model = model.from_pretrained("Rostlab/prot_albert", return_dict=False)

    if not pretrained_weights:
        custom_model = CustomModel(model,
                                   hidden_size=model_config.hidden_size,
                                   mid_token=int((configs['window_size'] + 1) / 2))
    else:
        custom_model = CustomModel(model,
                                   hidden_size=model_config.hidden_size,
                                   mid_token=int((configs['window_size'] + 1) / 2) + 1)
    # for parameter in custom_model.parameters():
    #     print(parameter.shape)

    # print('Number of parameters:', sum(p.numel() for p in custom_model.parameters() if p.requires_grad))

    custom_model = custom_model.to(device)
    return custom_model


def bert(tokenizer, configs, device, pretrained_weights):
    if pretrained_weights:
        max_len = configs['window_size'] + 2
        vocab_size = len(tokenizer)
    else:
        max_len = configs['window_size']
        vocab_size = len(tokenizer) + 1

    model_config = BertConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max_len,
        # embedding_size=768,
        # hidden_size=768,
        # num_attention_heads=8,
        # intermediate_size=1024,
        num_hidden_layers=8,
        return_dict=False,
        # classifier_activation=True
    )
    model = BertModel(model_config)

    # print('Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if pretrained_weights:
        print('no pretrained weights is considered')
        custom_model = None
    else:
        custom_model = CustomModel(model,
                                   hidden_size=model_config.hidden_size,
                                   mid_token=int((configs['window_size'] + 1) / 2))

    # for parameter in custom_model.parameters():
    #     print(parameter.shape)

    # print('Number of parameters:', sum(p.numel() for p in custom_model.parameters() if p.requires_grad))

    custom_model = custom_model.to(device)
    return custom_model


def tiny_bert(tokenizer, configs, device, pretrained_weights):
    if pretrained_weights:
        max_len = configs['window_size'] + 2
        vocab_size = len(tokenizer)
    else:
        max_len = configs['window_size']
        vocab_size = len(tokenizer) + 1

    model_config = BertConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max_len,
        # embedding_size=768,
        hidden_size=320,
        num_attention_heads=8,
        intermediate_size=768,
        num_hidden_layers=8,
        return_dict=False,
        # classifier_activation=True
    )
    model = BertModel(model_config)

    # print('Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if pretrained_weights:
        print('no pretrained weights is considered')
        custom_model = None
    else:
        custom_model = CustomModel(model,
                                   hidden_size=model_config.hidden_size,
                                   mid_token=int((configs['window_size'] + 1) / 2))

    # for parameter in custom_model.parameters():
    #     print(parameter.shape)

    # print('Number of parameters:', sum(p.numel() for p in custom_model.parameters() if p.requires_grad))

    custom_model = custom_model.to(device)
    return custom_model


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


def tabnet(tokenizer, configs, device, pretrained_weights):
    if pretrained_weights:
        max_len = configs['window_size'] + 2
        vocab_size = len(tokenizer)
    else:
        max_len = configs['window_size']
        vocab_size = len(tokenizer) + 1

    model = TabNetModel(vocab_size=vocab_size, window_size=int(configs['window_size']))

    # print('Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = model.to(device)
    return model


def prepare_model(device, configs, tokenizer, print_params=True):
    if configs['backbone'] == 'albert':
        model = albert(tokenizer, configs, device, pretrained_weights=configs['pretrained_weights'])
    elif configs['backbone'] == 'mobile_bert':
        model = mobile_bert(tokenizer, configs, device, pretrained_weights=configs['pretrained_weights'])
    elif configs['backbone'] == 'bert':
        model = bert(tokenizer, configs, device, pretrained_weights=configs['pretrained_weights'])
    elif configs['backbone'] == 'tiny_bert':
        model = tiny_bert(tokenizer, configs, device, pretrained_weights=configs['pretrained_weights'])
    elif configs['backbone'] == 'squeeze_bert':
        model = squeeze_bert(tokenizer, configs, device, pretrained_weights=configs['pretrained_weights'])
    elif configs['backbone'] == 'nystromformer':
        model = nystromformer(tokenizer, configs, device, pretrained_weights=configs['pretrained_weights'])
    elif configs['backbone'] == 'lstm':
        model = lstm(tokenizer, configs, device, pretrained_weights=configs['pretrained_weights'])
    elif configs['backbone'] == 'tabnet':
        model = tabnet(tokenizer, configs, device, pretrained_weights=configs['pretrained_weights'])
    else:
        print('wrong given backbone')
        model = None
        exit()
    if print_params:
        print('Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model
