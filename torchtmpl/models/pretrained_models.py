import torch
import torchvision.models as torch_models
import torch.nn as nn
import logging

class InceptionV3Model(nn.Module):
    def __init__(self,cfg,input_img_size,input_tab_size,num_classes):
        super(InceptionV3Model, self).__init__()
        self.model = torch_models.inception_v3(weights="DEFAULT")
        self.model.fc = nn.Linear(2048, num_classes)
        self.model.aux_logits = False
        self.model.eval()

    def forward(self, input_imgs):
        # Take only RGB image
        # x = input_imgs[:,3:,:,:]

        return self.model(input_imgs)

