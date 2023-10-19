import torch
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


class DNNModel(torch.nn.Module):
    def __init__(self, window_size, features):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.features = features
        if features['physiochemical'] and features['pssm'] and features['aac'] and features['dpc']:
            self.h_1 = torch.nn.Linear(16+42+20+400, 128)  # todo
        elif features['physiochemical'] and features['pssm'] and features['aac']:
            self.h_1 = torch.nn.Linear(16+42+20, 128)  # todo
        elif features['physiochemical'] and features['pssm']:
            self.h_1 = torch.nn.Linear(16+42, 128)  # todo
        elif features['pssm']:
            self.h_1 = torch.nn.Linear(42*window_size, 128)
        elif features['physiochemical']:
            self.h_1 = torch.nn.Linear(16*window_size, 128)
        elif features['aac']:
            self.h_1 = torch.nn.Linear(20*window_size, 128)
        elif features['dpc']:
            self.h_1 = torch.nn.Linear(400, 128)  # todo

        self.batchnorm_1 = torch.nn.BatchNorm1d(128)
        self.relu_1 = torch.nn.ReLU()

        self.h_2 = torch.nn.Linear(128, 64)
        self.batchnorm_2 = torch.nn.BatchNorm1d(64)
        self.relu_2 = torch.nn.ReLU()

        self.classifier = torch.nn.Linear(64, 2)

    def forward(self, sentence, features):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        if self.features['pssm']:
            feature = features['pssm']

        elif self.features['physiochemical']:
            feature = features['physiochemical']

        elif self.features['aac']:
            feature = features['aac']

        else:
            feature = features['dpc']

        h_1_out = self.h_1(torch.reshape(feature, (feature.shape[0], -1)))
        h_2_out = self.relu_2(self.batchnorm_2(self.h_2(self.relu_1(self.batchnorm_1(h_1_out)))))
        pred = self.classifier(h_2_out)
        return pred


class LSTMModel(torch.nn.Module):
    def __init__(self, vocab_size, mid_token, features, batchnorm=True):
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
        self.batchnorm = batchnorm
        self.features = features
        if self.batchnorm:
            if features['physiochemical'] and features['pssm'] and features['aac'] and features['dpc']:
                self.norm = torch.nn.BatchNorm1d(16+42+20+400)
            elif features['physiochemical'] and features['pssm'] and features['aac']:
                self.norm = torch.nn.BatchNorm1d(16+42+20)
            elif features['physiochemical'] and features['pssm']:
                self.norm = torch.nn.BatchNorm1d(16+42)
            elif features['pssm']:
                self.norm = torch.nn.BatchNorm1d(42)
            elif features['physiochemical']:
                self.norm = torch.nn.BatchNorm1d(16)
            elif features['aac']:
                self.norm = torch.nn.BatchNorm1d(20)
            elif features['dpc']:
                self.norm = torch.nn.BatchNorm1d(400)

        if features['pssm'] and features['physiochemical'] and features['aac'] and features['dpc']:
            self.lstm_1 = torch.nn.LSTM(input_size=256+42+16+20+400, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)
        elif features['pssm'] and features['physiochemical'] and features['aac']:
            self.lstm_1 = torch.nn.LSTM(input_size=256+42+16+20, hidden_size=32, num_layers=2, batch_first=True, bidirectional=True)
        elif features['pssm'] and features['physiochemical']:
            self.lstm_1 = torch.nn.LSTM(input_size=256+42+16, hidden_size=32, num_layers=2, batch_first=True, bidirectional=True)
        elif features['pssm']:
            self.lstm_1 = torch.nn.LSTM(input_size=256+42, hidden_size=32, num_layers=2, batch_first=True, bidirectional=True)
        elif features['physiochemical']:
            self.lstm_1 = torch.nn.LSTM(input_size=256+16, hidden_size=32, num_layers=2, batch_first=True, bidirectional=True)
        elif features['aac']:
            self.lstm_1 = torch.nn.LSTM(input_size=256+20, hidden_size=32, num_layers=2, batch_first=True, bidirectional=True)
        elif features['dpc']:
            self.lstm_1 = torch.nn.LSTM(input_size=256+400, hidden_size=32, num_layers=2, batch_first=True, bidirectional=True)

        self.classifier = torch.nn.Linear(64, 2)

    def forward(self, sentence, features):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        embeds = self.emb(sentence)
        if self.batchnorm:
            if self.features['physiochemical'] and self.features['pssm'] and self.features['aac'] and self.features['dpc']:
                physiochemical = features['physiochemical']
                pssm = features['pssm']
                aac = features['aac']
                dpc = features['dpc']
                concat_features = torch.concat((pssm, physiochemical, aac, dpc), dim=-1).permute(0, 2, 1)
                concat_features = self.norm(concat_features)
                concat_features = concat_features.permute(0, 2, 1)
                lstm_1_out, _ = self.lstm_1(torch.concat((embeds, concat_features), dim=-1))

            elif self.features['physiochemical'] and self.features['pssm'] and self.features['aac']:
                physiochemical = features['physiochemical']
                pssm = features['pssm']
                aac = features['aac']
                concat_features = torch.concat((pssm, physiochemical, aac), dim=-1).permute(0, 2, 1)
                concat_features = self.norm(concat_features)
                concat_features = concat_features.permute(0, 2, 1)
                lstm_1_out, _ = self.lstm_1(torch.concat((embeds, concat_features), dim=-1))

            elif self.features['physiochemical'] and self.features['pssm']:
                physiochemical = features['physiochemical']
                pssm = features['pssm']
                concat_features = torch.concat((pssm, physiochemical), dim=-1).permute(0, 2, 1)
                concat_features = self.norm(concat_features)
                concat_features = concat_features.permute(0, 2, 1)
                lstm_1_out, _ = self.lstm_1(torch.concat((embeds, concat_features), dim=-1))

            elif self.features['pssm']:
                pssm = features['pssm'].permute(0, 2, 1)
                pssm = self.norm(pssm)
                pssm = pssm.permute(0, 2, 1)
                lstm_1_out, _ = self.lstm_1(torch.concat((embeds, pssm), dim=-1))

            elif self.features['physiochemical']:
                physiochemical = features['physiochemical'].permute(0, 2, 1)
                physiochemical = self.norm(physiochemical)
                physiochemical = physiochemical.permute(0, 2, 1)
                lstm_1_out, _ = self.lstm_1(torch.concat((embeds, physiochemical), dim=-1))

            elif self.features['aac']:
                aac = features['aac'].permute(0, 2, 1)
                aac = self.norm(aac)
                aac = aac.permute(0, 2, 1)
                lstm_1_out, _ = self.lstm_1(torch.concat((embeds, aac), dim=-1))

            elif self.features['dpc']:
                dpc = features['dpc'].permute(0, 2, 1)
                dpc = self.norm(dpc)
                dpc = dpc.permute(0, 2, 1)
                lstm_1_out, _ = self.lstm_1(torch.concat((embeds, dpc), dim=-1))

            else:
                lstm_1_out, _ = self.lstm_1(embeds)
        else:
            if self.features['physiochemical'] and self.features['pssm'] and self.features['aac'] and self.features['dpc']:
                physiochemical = features['physiochemical']
                pssm = features['pssm']
                aac = features['aac']
                dpc = features['dpc']
                lstm_1_out, _ = self.lstm_1(torch.concat((embeds, pssm, physiochemical, aac, dpc), dim=-1))

            elif self.features['physiochemical'] and self.features['pssm'] and self.features['aac']:
                physiochemical = features['physiochemical']
                pssm = features['pssm']
                aac = features['aac']
                lstm_1_out, _ = self.lstm_1(torch.concat((embeds, pssm, physiochemical, aac), dim=-1))

            elif self.features['physiochemical'] and self.features['pssm']:
                physiochemical = features['physiochemical']
                pssm = features['pssm']
                lstm_1_out, _ = self.lstm_1(torch.concat((embeds, pssm, physiochemical), dim=-1))

            elif self.features['pssm']:
                pssm = features['pssm']
                lstm_1_out, _ = self.lstm_1(torch.concat((embeds, pssm), dim=-1))

            elif self.features['physiochemical']:
                physiochemical = features['physiochemical']
                lstm_1_out, _ = self.lstm_1(torch.concat((embeds, physiochemical), dim=-1))

            elif self.features['aac']:
                aac = features['aac']
                lstm_1_out, _ = self.lstm_1(torch.concat((embeds, aac), dim=-1))

            elif self.features['dpc']:
                dpc = features['dpc']
                lstm_1_out, _ = self.lstm_1(torch.concat((embeds, dpc), dim=-1))

            else:
                lstm_1_out, _ = self.lstm_1(embeds)

        pred = self.classifier(lstm_1_out[:, self.mid_token, :])
        return pred


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


def lstm(tokenizer, configs, device, pretrained_weights):
    if pretrained_weights:
        max_len = configs['window_size'] + 2
        vocab_size = len(tokenizer)
    else:
        max_len = configs['window_size']
        vocab_size = len(tokenizer) + 1

    model = LSTMModel(vocab_size=vocab_size, mid_token=int((configs['window_size'] + 1) / 2),
                      features=configs['features'])

    # print('Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = model.to(device)
    return model


def dnn(tokenizer, configs, device, pretrained_weights):
    if pretrained_weights:
        max_len = configs['window_size'] + 2
        vocab_size = len(tokenizer)
    else:
        max_len = configs['window_size']
        vocab_size = len(tokenizer) + 1

    model = DNNModel(window_size=configs['window_size'], features=configs['features'])

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
    elif configs['backbone'] == 'squeeze_bert':
        model = squeeze_bert(tokenizer, configs, device, pretrained_weights=configs['pretrained_weights'])
    elif configs['backbone'] == 'nystroformer':
        model = nystromformer(tokenizer, configs, device, pretrained_weights=configs['pretrained_weights'])
    elif configs['backbone'] == 'lstm':
        model = lstm(tokenizer, configs, device, pretrained_weights=configs['pretrained_weights'])
    elif configs['backbone'] == 'dnn':
        model = dnn(tokenizer, configs, device, pretrained_weights=configs['pretrained_weights'])
    else:
        print('wrong given backbone')
        model = None
        exit()
    if print_params:
        print('Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model
