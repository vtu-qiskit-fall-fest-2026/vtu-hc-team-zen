import torch
import torch.nn as nn
import torch.optim as optim
from src.main import HybridUNet

def train_model(model, dataloader, epochs=2, lr=0.01):
    """Basic training loop for the Hybrid Quantum-Classical U-Net."""
    # Binary Cross Entropy Loss (standard for image segmentation/masking)
    criterion = nn.BCELoss() 
    # Adam optimizer will update both classical weights and quantum parameters
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass (Classical -> Quantum -> Classical)
            outputs = model(inputs)
            
            # Calculate loss against the ground-truth mask
            loss = criterion(outputs, targets)
            
            # Backward pass (Backpropagation through the quantum circuit)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            print(f"Epoch {epoch+1} | Batch {batch_idx+1} | Loss: {loss.item():.4f}")
            
        print(f"--- Epoch {epoch+1} Complete | Average Loss: {epoch_loss/len(dataloader):.4f} ---")

if __name__ == "__main__":
    print("Setting up dummy dataset to test the training loop...")
    
    # Dummy inputs: 4 images, 3 channels (RGB), 32x32 pixels
    dummy_images = torch.rand(4, 3, 32, 32)
    # Dummy targets: 4 masks, 1 channel, 32x32 pixels (binary 0 or 1)
    dummy_masks = torch.randint(0, 2, (4, 1, 32, 32), dtype=torch.float32)
    
    # Pack into a basic dataloader list
    dummy_dataloader = [(dummy_images, dummy_masks)]
    
    print("Initializing Model...")
    model = HybridUNet()
    
    print("Starting Hybrid Training (This may take a moment to calculate quantum gradients)...")
    train_model(model, dummy_dataloader, epochs=2)
