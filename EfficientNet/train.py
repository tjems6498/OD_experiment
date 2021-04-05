import albumentations as A
import argparse
import torch
import torch.optim as optim
from efficientnet_model import EfficientNet
from dataset import Person
from utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm


torch.backends.cudnn.benchmark = False
'''
내장된 cudnn 자동 튜너를 활성화하여, 하드웨어에 맞게 사용할 최상의 알고리즘(텐서 크기나 conv 연산에 맞게?)을 찾는다.
입력 이미지 크기가 자주 변하지 않는다면, 초기 시간이 소요되지만 일반적으로 더 빠른 런타임의 효과를 볼 수 있다.
그러나, 입력 이미지 크기가 반복될 때마다 변경된다면 런타임성능이 오히려 저하될 수 있다.
'''

def train(train_loader, model, optimizer, criterion, scaler, device):
    model.train()
    loop = tqdm(train_loader, leave=True)  # leave : 반복이 종료될때 진행률 흔적을 유지 defailt=True
    losses = []
    total = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(loop):
        inputs = inputs.to(device)

        with torch.cuda.amp.autocast():
            out = model(inputs)
            loss = criterion(out, targets)
        losses.append(loss.item())
        _, predicted = out.max(1)
        correct += (predicted == targets).sum().item()
        total += targets.shape[0]

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_loss = sum(losses) / len(losses)
        mean_accuracy = correct / total * 100
        loop.set_postfix(loss=f'{mean_loss:.4f}', accuracy=f'{mean_accuracy:.4f}')

    return mean_accuracy


def main(args, device):
    model = EfficientNet(args.version, args.num_classes, args.weights).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()  # FP16

    train_dataset = Person(args.data_folder, args.classes, transform=get_augmentation('train', width=224, height=224))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    if args.load_model:
        load_checkpoint(
            "checkpoint.pth.tar", model, optimizer, args.LEARNING_RATE, device
        )
    top_acc = 0
    for epoch in range(args.epochs):
        accuracy = train(train_loader, model, optimizer, criterion, scaler, device=device)

        if top_acc < accuracy:
            if args.save_model:
                save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='b0', help='can use b0 to b7')
    parser.add_argument('--batch-size', type=int, default=2, help='total batch size for all GPUs')
    parser.add_argument('--num-classes', type=int, default=3, help='number of classes(man, woman, kid)')
    parser.add_argument('--epochs', type=int, defualt=100)
    parser.add_argument('--data-folder', type=str, default='E:\\Computer Vision\\data\\project\\person')
    parser.add_argument('--lr', type=float, default=3e-5, help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='l2 normalization')
    parser.add_argument('--classes', type=dict, default={'kid': 0, 'man': 1, 'woman': 2}, help='class info')
    parser.add_argument('--load-model', type=bool, defualt=False, help='load trained model')
    parser.add_argument('--save-model', type=bool, defualt=True, help='save model')
    parser.add_argument('--weights', type=bool, defualt=False, help='use pretrained model weights')

    args = parser.parse_args()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(args, DEVICE)
