import os
import torch
import torch.nn as nn
import torchvision.models as models
# from dataset import dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
import timm
# from simclr import SimCLR_cls

def parse_option():
    parser = argparse.ArgumentParser('ViT-S/16 for Classification')
    # Training setting
    parser.add_argument('--batch_size', type=int, default=128, 
                    help='batch_size')
    parser.add_argument('--epoch', type=int, default=30,
                    help='number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-3,
                    help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--device', type=str, default='cuda:1')
    # dataset & model
    parser.add_argument('--seed', type=int, default=850011,
                    help='seed for initializing training')
    parser.add_argument('--root', type=str, default='./dataset/',
                    help='dataset path')
    # parser.add_argument('--pretrained',default=False, action="store_true")
    parser.add_argument('--save_dir', type=str, default='./save',
                    help='path to save models')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pretrained', default=False, action="store_true")


    args = parser.parse_args()

    return args


# import 
def main():

    args = parse_option()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    os.makedirs(args.save_dir, exist_ok=True)
    save = os.path.join(args.save_dir, f'train_vit_{args.pretrained}_{args.lr:.1e}')
    os.makedirs(save, exist_ok=True)

    
    batch_size = args.batch_size
    device = args.device if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CIFAR100(args.root, transform=transform,
                            download=True, train=True)

    val_dataset = CIFAR100(args.root, transform=transform,
                        download=True, train=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size, pin_memory=True,
                              num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size, pin_memory=True,
                            num_workers=args.num_workers, shuffle=False)
    

    log_dir = './train_logs'
    writer = SummaryWriter(os.path.join(log_dir ,datetime.now().strftime("%m%d-%H%M")+f'_train_vit_{args.lr:.1e}_{args.pretrained}'))
    
    model = timm.create_model('vit_small_patch16_224', pretrained=args.pretrained, num_classes=100).to(device)
    # import pdb;pdb.set_trace() # 分类层是model.heads
    # optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    # model.to(device)
    if args.pretrained:
        # model = models.resnet34(pretrained=True)
        # model.fc = nn.Linear(in_features=512, out_features=200, bias=True)
        params_fc = [p for p in model.head.parameters()] 
        params_others = [p for p in model.parameters() if id(p) not in [id(p) for p in params_fc]]
        optimizer = torch.optim.Adam([
                {'params': params_fc, 'lr': args.lr},
                {'params': params_others, 'lr': args.lr*0.1},
            ], weight_decay=args.weight_decay,)
    else:
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    epoch = args.epoch
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=1e-8,last_epoch=-1) # update in every step instead of epoch
    criterion = nn.CrossEntropyLoss().to(device)

    
    best_acc = 0
    for ep in range(1,epoch+1):
        print(f'Training for epoch {ep}')
        # import pdb;pdb.set_trace()
        # writer.add_scalar('lr',scheduler.get_last_lr()[0],ep)
        running_loss = 0.
        for batch in tqdm(train_loader,total=len(train_loader)):
            x,y=batch
            x = x.to(device)
            y = y.to(device)
            # import pdb;pdb.set_trace()

            optimizer.zero_grad()
            y_pre = model(x)
            loss = criterion(y_pre, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        if ep > 10:  # warmup for the first 10 epochs
            scheduler.step()
        print(f'Train loss: {running_loss}')
        # scheduler.step()
        
        print(f'Validating for epoch {ep}')
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for data in tqdm(val_loader, total=len(val_loader)):
                images, y = data
                images, y = images.to(device), y.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, y).item()

                predicted = torch.argmax(outputs, 1)
                
                # labels = torch.argmax(y, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            accuracy = 100 * correct / total
        print(f'Val loss: {val_loss}')
        print(f'Val Accuracy:{accuracy}')
        
        writer.add_scalar('Loss/train',running_loss, ep)
        writer.add_scalar('Loss/validation',val_loss, ep)
        writer.add_scalar('Accuracy', accuracy, ep)

        if accuracy > best_acc:
            torch.save(model.state_dict(), os.path.join(save,f'ep_{ep}_best_{accuracy}.pth'))
            best_acc = accuracy
    
    torch.save(model.state_dict(), os.path.join(save, 'last.pth'))
    print(f'best acc: {best_acc}')
        
if __name__=="__main__":
    main()    

    