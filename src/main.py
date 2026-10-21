import torch
import torch.nn as nn
from src.circuit import create_quantum_bottleneck

class HybridUNet(nn.Module):
    def __init__(self):
        super(HybridUNet, self).__init__()
        # 1. Classical Encoder (Extracts spatial features)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 2. Quantum Bottleneck (Reduces parameter dimensionality)
        # Flattening 8 channels of 16x16 pixels to connect to our 4 qubits
        self.fc_in = nn.Linear(8 * 16 * 16, 4) 
        self.quantum_layer = create_quantum_bottleneck(num_qubits=4)
        self.fc_out = nn.Linear(4, 8 * 16 * 16)
        
        # 3. Classical Decoder (Reconstructs the segmentation mask)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid() # Outputs values between 0 and 1 for road probability
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # Pass through classical encoder
        x = self.encoder(x)
        
        # Reshape and process through Quantum layer
        x_flat = x.view(batch_size, -1)
        x_q_in = self.fc_in(x_flat)
        x_q_out = self.quantum_layer(x_q_in)
        
        # Reshape and pass through classical decoder
        x_unflat = self.fc_out(x_q_out).view(batch_size, 8, 16, 16)
        out = self.decoder(x_unflat)
        
        return out

if __name__ == "__main__":
    print("Initializing Hybrid Quantum-Classical U-Net...")
    model = HybridUNet()
    
    # Simulating a batch of 2 RGB satellite images (32x32 pixels)
    dummy_satellite_images = torch.rand(2, 3, 32, 32)
    
    print("Passing satellite imagery through the Quantum Model...")
    output_mask = model(dummy_satellite_images)
    
    print(f"Input shape (Batch, Channels, H, W): {dummy_satellite_images.shape}")
    print(f"Output mask shape: {output_mask.shape}")
    print("Hybrid Model Forward Pass Successful!")
