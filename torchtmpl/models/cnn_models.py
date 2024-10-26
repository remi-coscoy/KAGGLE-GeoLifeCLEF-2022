# coding: utf-8

# Standard imports
from functools import reduce
import operator

# External imports
import torch
import torch.nn as nn


def conv_relu_bn(cin, cout):
    return [
        nn.Dropout2d(0.1),
        nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
    ]


def conv_down(cin, cout):
    return [
        nn.Dropout2d(0.1),
        nn.Conv2d(cin, cout, kernel_size=2, stride=2, padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
    ]


def VanillaCNN(cfg, input_img_size, input_tab_size, num_classes):

    layers = []
    cin = input_img_size[0]
    cout = 16
    for i in range(cfg["num_layers"]):
        layers.extend(conv_relu_bn(cin, cout))
        layers.extend(conv_relu_bn(cout, cout))
        if i < cfg["num_layers"] - 2:
            layers.extend(conv_down(cout, 2 * cout))
            cin = 2 * cout
            cout = 2 * cout
        else:
            layers.extend(conv_down(cout, cout))
        # layers.extend(conv_down(cout, 2 * cout))
        
    conv_model = nn.Sequential(*layers)

    # Compute the output size of the convolutional part
    probing_tensor = torch.zeros((1,) + input_img_size)
    out_cnn = conv_model(probing_tensor)  # B, K, H, W
    num_features = reduce(operator.mul, out_cnn.shape[1:], 1)
    out_layers = [nn.Flatten(start_dim=1), nn.Linear(num_features, num_classes)]
    
    # multimodal model not using tabular data
    return nn.Sequential(conv_model, *out_layers)

class VanillaCNNmodel(nn.Module):
    def __init__(self, cfg, input_img_size, input_tab_size, num_classes):
        super(VanillaCNNmodel, self).__init__()

        layers = []
        cin = input_img_size[0]
        cout = 16
        for i in range(cfg["num_layers"]):
            layers.extend(conv_relu_bn(cin, cout))
            layers.extend(conv_relu_bn(cout, cout))
            if i < cfg["num_layers"] - 2:
                layers.extend(conv_down(cout, 2 * cout))
                cin = 2 * cout
                cout = 2 * cout
            else:
                layers.extend(conv_down(cout, cout))
            # layers.extend(conv_down(cout, 2 * cout))
            
        conv_model = nn.Sequential(*layers)

        # Compute the output size of the convolutional part
        probing_tensor = torch.zeros((1,) + input_img_size)
        out_cnn = conv_model(probing_tensor)  # B, K, H, W
        num_features = reduce(operator.mul, out_cnn.shape[1:], 1)
        out_layers = [nn.Flatten(start_dim=1), nn.Linear(num_features, num_classes)]
        
        self.model = nn.Sequential(conv_model, *out_layers)
    
    def forward(self, input_imgs, input_tab):
        
        x = self.model(input_imgs)
        return x
    
class MultimodalCNN(nn.Module):
    def __init__(self, cfg, input_img_size, input_tab_size, num_classes):
        super(MultimodalCNN, self).__init__()

        layers = []
        cin = input_img_size[0]
        cout = 16
        for i in range(cfg["num_layers"]):
            layers.extend(conv_relu_bn(cin, cout))
            layers.extend(conv_relu_bn(cout, cout))
            if i < cfg["num_layers"] - 2:
                layers.extend(conv_down(cout, 2 * cout))
                cin = 2 * cout
                cout = 2 * cout
            else:
                layers.extend(conv_down(cout, cout))
            # layers.extend(conv_down(cout, 2 * cout))
            
        self.conv_model = nn.Sequential(*layers)

        # Compute the output size of the convolutional part
        probing_tensor = torch.zeros((1,) + input_img_size)
        out_cnn = self.conv_model(probing_tensor)  # B, K, H, W
        num_features = reduce(operator.mul, out_cnn.shape[1:], 1)
        self.flat = nn.Flatten(start_dim=1)
        self.lin1 = nn.Linear(num_features + input_tab_size[0], num_classes)
        self.lin2 = nn.Linear(num_classes, num_classes)
            
    def forward(self, input_imgs, input_tab):
        x = self.conv_model(input_imgs)
        x = self.flat(x)
        x = torch.cat((x, input_tab), dim=1)
        x = self.lin1(x)
        x = self.lin2(x)
        return x
    
class MultimodalCNN2(nn.Module):
    def __init__(self, cfg, input_img_size, input_tab_size, num_classes):
        super(MultimodalCNN2, self).__init__()

        layers = []
        cin = input_img_size[0]-1 + cfg["embedding_landcover_size"]
        
        
        cout = 16
        for i in range(cfg["num_layers"]):
            layers.extend(conv_relu_bn(cin, cout))
            layers.extend(conv_relu_bn(cout, cout))
            if i < cfg["num_layers"] - 2:
                layers.extend(conv_down(cout, 2 * cout))
                cin = 2 * cout
                cout = 2 * cout
            else:
                layers.extend(conv_down(cout, cout))
            # layers.extend(conv_down(cout, 2 * cout))
            
        self.conv_model = nn.Sequential(*layers)

        # Compute the output size of the convolutional part
        probing_tensor = torch.zeros((1,) + (input_img_size[0]-1+ cfg["embedding_landcover_size"],input_img_size[1],input_img_size[2]))
        out_cnn = self.conv_model(probing_tensor)
        num_features = reduce(operator.mul, out_cnn.shape[1:], 1)
        self.flat = nn.Flatten(start_dim=1)
        self.lin1 = nn.Linear(num_features + input_tab_size[0], num_classes)
        self.lin2 = nn.Linear(num_classes, num_classes)
        
        self.embed = nn.Embedding(34,cfg["embedding_landcover_size"])
        
        self.batchnorm_lin = nn.BatchNorm1d(num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
            
    def forward(self, input_imgs, input_tab):
        imgs = input_imgs[:,:-1,:,:]
        landcover = input_imgs[:,-1,:,:].long()
        landcover = self.embed(landcover)
        landcover = landcover.permute(0,3,1,2)
        x = torch.cat((imgs,landcover),dim=1)
        x = self.conv_model(x)
        x = self.flat(x)
        x = torch.cat((x, input_tab), dim=1)
        x = self.dropout(x)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.batchnorm_lin(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return x