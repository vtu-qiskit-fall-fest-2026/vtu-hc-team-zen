import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.main import HybridUNet
from src.dataset import SatelliteRoadDataset

def train_model(model, dataloader, epochs=2, lr=0.01, device='cpu'):
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            print(f"Epoch {epoch+1} | Batch {batch_idx+1} | Loss: {loss.item():.4f}")
            
        print(f"--- Epoch {epoch+1} Complete | Average Loss: {epoch_loss/len(dataloader):.4f} ---")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    dataset = SatelliteRoadDataset(images_dir="data/images", masks_dir="data/masks", image_size=32)
    
    if len(dataset) == 0:
        print("WAITING FOR DATA: Folders are empty!")
    else:
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        print("Initializing Hybrid Model...")
        model = HybridUNet().to(device)
        print(f"Starting Hybrid Training on {len(dataset)} real ISRO images...")
        train_model(model, dataloader, epochs=3, device=device)
        
        # Save the trained model weights
        torch.save(model.state_dict(), "hybrid_unet_model.pth")
        print("Model weights saved to hybrid_unet_model.pth")
