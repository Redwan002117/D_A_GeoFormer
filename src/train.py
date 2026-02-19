import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SpaceNet8Dataset
from model import DualAxisGeoFormer
from loss import TverskyLoss
from augmentation import CopyPasteAugmentation
import argparse

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
    
    # Augmentation
    augmentor = CopyPasteAugmentation(prob=0.5)

    # Loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        if train_loader:
            for i, batch in enumerate(train_loader):
                # Unpack
                pre_img = batch['pre_img'].to(device)
                post_img = batch['post_img'].to(device)
                mask = batch['mask'].to(device)
                
                # Forward
                outputs = model(pre_img, post_img)
                
                # Loss
                loss = criterion(outputs, mask) # Ensure mask shape matches (B, H, W)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if i % 10 == 0:
                    print(f"Epoch [{epoch}/{args.epochs}], Step [{i}], Loss: {loss.item():.4f}")
        else:
            # Dummy loop
            print("Running dummy loop...")
            pre_img = torch.randn(2, 3, 256, 256).to(device)
            post_img = torch.randn(2, 3, 256, 256).to(device)
            mask = torch.randint(0, 4, (2, 256, 256)).to(device)
            outputs = model(pre_img, post_img)
            loss = criterion(outputs, mask)
            loss.backward()
            print(f"Dummy Loss: {loss.item()}")
            break

        print(f"Epoch {epoch} finished. Avg Loss: {epoch_loss / max(len(train_loader), 1)}")

        # Validation (omitted for brevity)
        
        # Save checkpoint
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/SN8')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    train(args)
