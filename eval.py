import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

import argparse
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
import timm

def parse_option():
    parser = argparse.ArgumentParser('CutMix for Classification')
    parser.add_argument('--batch_size', type=int, default=128, 
                    help='batch_size')
    # dataset & model
    parser.add_argument('--seed', type=int, default=850011,
                    help='seed for initializing training')
    parser.add_argument('--root', type=str, default='./dataset/',
                    help='dataset path')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--file', type=str, default='./save/train_vit_True_1.0e-04/ep_27_best_89.99.pth',
                    help='model pth path')
    parser.add_argument('--model', type=str, default='vit',
                        choices=['vit','rn34'])
    parser.add_argument('--cutmix', default=False, action="store_true")
    args = parser.parse_args()

    return args


# import 
def main():

    args = parse_option()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])



    val_dataset = CIFAR100(args.root, transform=transform,
                        download=True, train=False)


    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size, pin_memory=True,
                            num_workers=args.num_workers, shuffle=False)
    
    if args.model == 'vit':
        model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=100).to(device)

    elif args.model == 'rn34':
        model = timm.create_model('resnet34', pretrained=True, num_classes=100).to(device)
    
    else:
        raise Exception('unknown network architecture: {}'.format(args.model))
    
    if args.cutmix:
        model.load_state_dict(torch.load(args.file)['state_dict'])
    else:
        model.load_state_dict(torch.load(args.file))



    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(val_loader, total=len(val_loader)):
            images, y = data
            images, y = images.to(device), y.to(device)
            outputs = model(images)

            predicted = torch.argmax(outputs, 1)
            
            # labels = torch.argmax(y, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        accuracy = 100 * correct / total

    print(f'Val Accuracy:{accuracy}')
        

if __name__=="__main__":
    main()    

    