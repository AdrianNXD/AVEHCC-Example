import torch
import torch.nn as nn
from torchvision import models


def mobilenet_v3_large(num_classes:int):

    in_channels = 1

    model = models.mobilenet_v3_large(weights="DEFAULT")

    # Adjust the first layer: RGB -> grayscale
    old_conv = model.features[0][0]

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode
    )

    # Calculate averaged weights from the RGB channels
    with torch.no_grad():

        if in_channels == 1:
            new_conv.weight.copy_(
                old_conv.weight.mean(dim=1, keepdim=True)
            )
        else:
            nn.init.kaiming_normal_(
                new_conv.weight,
                mode="fan_out",
                nonlinearity="relu"
            )

        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    model.features[0][0] = new_conv

    # Reduce the number of output classes to num_classes
    old_classifier = model.classifier[3]

    model.classifier[3] = nn.Linear(
        in_features=old_classifier.in_features,
        out_features=num_classes,
        bias=old_classifier.bias is not None
    )

    # Initialize the new layer with random weights
    nn.init.normal_(
        model.classifier[3].weight,
        mean=0.0,
        std=0.01
    )

    if model.classifier[3].bias is not None:
        nn.init.constant_(
            model.classifier[3].bias,
            0.0
        )

    return model