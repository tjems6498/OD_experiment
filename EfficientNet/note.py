import torch
from efficientnet_pytorch import EfficientNet
from efficientnet_model import EfficientNet as my_model


model = my_model(
    version='b0',
    num_classes=3,
)

state_dict = model.state_dict()
param_names = list(state_dict.keys())
# print(param_names)
print(state_dict['features.17.cnn.weight'])

pretrained_state_dict = EfficientNet.from_pretrained('efficientnet-b0').state_dict()
pretrained_param_names = list(pretrained_state_dict.keys())
# print(pretrained_param_names)

for i, param in enumerate(param_names[:-2]):
    state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

print(state_dict['features.17.cnn.weight'])

