import torch.nn as nn
from torchvision import transforms, models

from WSDAN import WSDAN

def init_model(model_idx, num_classes, use_pretrained):
    """
    model_idx:
        0 -> resnext
        1 -> resnet152
        2 -> resnet50
        3 -> resnet101
        4 -> WSDAN: Weakly Supervised Data Augmentation Network
    """
    model_ft = None
    input_size = 0
    valid_model_names = ['resnet101', 'resnet152', 'resnet50', 'resnext']
    
    if model_idx == 0:
        model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
    elif model_idx == 1:
        model_ft = models.resnet152(pretrained=use_pretrained)
    elif model_idx == 2:
        model_ft = models.resnet50(pretrained=use_pretrained)
    elif model_idx == 3:
        model_ft = models.resnet101(pretrained=use_pretrained)
    elif model_idx == 4:
        model = WSDAN(num_classes=num_classes, M=config.num_attentions, net=config.net, pretrained=True)
    else:
        print('Invalid model idx, exiting. . .') 
        exit();

#     set_parameters_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224

    return model_ft, input_size
