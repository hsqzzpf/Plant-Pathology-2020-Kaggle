from torchvision import transforms, models
import torch.nn as nn


def init_model(model_name, num_classes, features_extract, use_pretrained):
    model_ft = None
    input_size = 0
    valid_model_names = ['resnet101', 'resnet152', 'resnet50', 'resnext']
    
    if model_name not in valid_model_names:
        print('Invalid model name, exiting. . .') 
        exit();
    elif model_name == 'resnext':
        model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
    elif model_name == 'resnet152':
        model_ft = models.resnet152(pretrained=use_pretrained)
    elif model_name == 'resnet50':
        model_ft = models.resnet50(pretrained=use_pretrained)
    elif model_name == 'resnet101':
        model_ft = models.resnet101(pretrained=use_pretrained)

#     set_parameters_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224

    return model_ft, input_size
