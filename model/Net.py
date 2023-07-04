import torch.nn.functional as F
import torchvision.models as models
import torch
import torch.nn.modules as nn
import pdb


class ResNet(nn.Module):
    def __init__(self, config, hash_bit, label_size, pretrained=True):
        super(ResNet, self).__init__()
        model_resnet = models.resnet50(pretrained=pretrained)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layer = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                           self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        self.tanh = nn.Tanh()
        self.label_linear = nn.Linear(label_size, hash_bit)
        if config['without_BN']:
            self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
            self.hash_layer.weight.data.normal_(0, 0.01)
            self.hash_layer.bias.data.fill_(0.0)
        else:
            self.layer_hash = nn.Linear(model_resnet.fc.in_features, hash_bit)
            self.layer_hash.weight.data.normal_(0, 0.01)
            self.layer_hash.bias.data.fill_(0.0)
            self.hash_layer = nn.Sequential(self.layer_hash, nn.BatchNorm1d(hash_bit, momentum=0.1))

    def forward(self, x, T, label_vectors):

        feat = self.feature_layer(x)
        feat = feat.view(feat.shape[0], -1)
        x = self.hash_layer(feat)
        x = self.tanh(x)
        
        return x, feat

class MoCo(nn.Module):
    def __init__(self, config, hash_bit, label_size, pretrained=True):
        super(MoCo, self).__init__()
        self.m = config['mome']
        self.encoder_q = ResNet(config, hash_bit, label_size, pretrained)
        self.encoder_k = ResNet(config, hash_bit, label_size, pretrained)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, x, T, label_vectors):
        encode_x, _ = self.encoder_q(x, T, label_vectors)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            encode_x2, _ = self.encoder_k(x, T, label_vectors)
        return encode_x, encode_x2

