import cv2
import torch
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

def get_augmentation(phase, width=224, height=224):
    list_transforms = []
    if phase == 'train':
        list_transforms.extend(
            [
                A.Resize(width=600, height=600),
                # A.RandomCrop(width=width, height=height),
                A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
                A.OneOf([
                    A.Blur(blur_limit=3, p=0.5),
                    A.ColorJitter(p=0.5),
                ], p=1.0),
                A.Normalize(
                    mean=[0, 0, 0],
                    std=[1, 1, 1],
                    max_pixel_value=255,
                ),
                ToTensorV2(),
            ]
        )
    elif phase == 'test' or phase == 'valid':
        list_transforms.extend(
            [
                A.Resize(width=width, height=height),
                A.Normalize(
                    mean=[0, 0, 0],
                    std=[1, 1, 1],
                    max_pixel_value=255,
                ),
                ToTensorV2(),
            ]
        )

    return A.Compose(list_transforms)



def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        'state_dict': model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)





