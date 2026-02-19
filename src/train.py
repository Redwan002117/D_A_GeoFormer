import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SpaceNet8Dataset
from model import DualAxisGeoFormer
from loss import TverskyLoss
from augmentation import CopyPasteAugmentation
import argparse
import time

def train(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    # Assuming data exists or using dummy
    # For now, let's wrap in try-except to handle missing data gracefully during dev
    try:
        train_dataset = SpaceNet8Dataset(args.data_dir, split='train')
        val_dataset = SpaceNet8Dataset(args.data_dir, split='val')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    except Exception as e:
        print(f"Dataset init failed: {e}. Using dummy tensors for verification loop if needed.")
        train_loader = None

    # Model
    model = DualAxisGeoFormer(num_classes=4).to(device)
    
    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = TverskyLoss(alpha=0.3, beta=0.7)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Augmentation
    augmentor = CopyPasteAugmentation(prob=0.5)

    print("Starting training...")
    min_loss = float('inf')
    
    # Loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} | LR: {current_lr:.2e}")
        
        if train_loader:
            for i, batch in enumerate(train_loader):
                # Unpack
                pre_img = batch['pre_img'].to(device)
                post_img = batch['post_img'].to(device)
                mask = batch['mask'].to(device)
                
                # Forward
                outputs = model(pre_img, post_img)
                
                # Loss
                loss = criterion(outputs, mask)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if i % 10 == 0:
                    print(f"  Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
            # Step Scheduler
            scheduler.step()
        else:
            # Dummy loop
            time.sleep(0.1)
            pass

        avg_loss = epoch_loss / max(len(train_loader), 1) if train_loader else 0
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        is_best = avg_loss < min_loss
        if is_best:
            min_loss = avg_loss
            print(f"New best model found! Loss: {min_loss:.4f}")
            torch.save(model.state_dict(), "best_model.pth")
            
        if args.save_all:
             torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")
        
        # Always save last
        torch.save(model.state_dict(), "last_model.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/SN8')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_all', action='store_true', help='Save checkpoint every epoch')
    args = parser.parse_args()
    train(args)
