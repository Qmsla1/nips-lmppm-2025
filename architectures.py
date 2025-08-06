import torch
import torch.nn as nn
import torch.nn.functional as F

# model = UNet(in_channels=1, out_channels=1)
# print(model)
#
# # Dummy input
# x = torch.randn(1, 1, 128, 128)  # Batch size 1, single-channel image, 128x128 resolution
# output = model(x)

import torch
import torch.nn as nn


class FaceAutoencoder120_3skips(nn.Module):
    def __init__(self, bottleneck_size=1024):
        super(FaceAutoencoder120_3skips, self).__init__()
        
        # Define encoder blocks individually to access intermediate outputs for skip connections
        # First encoder block
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 120x120 -> 60x60
        
        # Second encoder block
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 60x60 -> 30x30
        
        # Third encoder block
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 30x30 -> 15x15
        
        # Fourth encoder block
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 15x15 -> 7x7 (approx)
        
        # Flattened bottleneck
        self.flatten = nn.Flatten()
        
        # Calculate size after encoding convolutional layers
        # With 120x120 input, after 4 MaxPool layers (dividing by 2^4 = 16)
        # We get 7.5x7.5 which becomes 7x7
        self.feature_size = 256 * 7 * 7
        
        # Dense bottleneck layer - LARGER bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(self.feature_size, bottleneck_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder begins with dense layer back to pre-flatten size
        self.decoder_dense = nn.Sequential(
            nn.Linear(bottleneck_size, self.feature_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Define decoder blocks with transposed convolutions for better upsampling
        # First decoder block
        self.upconv4 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),  # 512 due to skip connection
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Second decoder block
        self.upconv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),  # 256 due to skip connection
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Third decoder block - WITH skip connection from e2
        self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # 128 due to skip connection
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Fourth decoder block - NO skip connection
        self.upconv1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # No skip, so input channels = 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Additional handling for odd dimensions
        self.final_upconv = nn.Upsample(size=(120, 120), mode='bilinear', align_corners=True)
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )

    def encoder(self, x):
        # Store intermediate outputs for skip connections
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        flat = self.flatten(p4)
        bottleneck = self.bottleneck(flat)
        
        # Store e2, e3 and e4 for skip connections
        return bottleneck, (e2, e3, e4)
    
    def decoder(self, x, skip_features):
        e2, e3, e4 = skip_features
        
        x = self.decoder_dense(x)
        x = x.view(-1, 256, 7, 7)  # Reshape to match encoder output dimensions
        
        # Use skip connection with e4
        x = self.upconv4(x)
        # Handle potential size mismatch
        if x.size() != e4.size():
            x = nn.functional.interpolate(x, size=e4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, e4], dim=1)  # Skip connection with e4
        x = self.dec4(x)
        
        # Use skip connection with e3
        x = self.upconv3(x)
        if x.size() != e3.size():
            x = nn.functional.interpolate(x, size=e3.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, e3], dim=1)  # Skip connection with e3
        x = self.dec3(x)
        
        # Use skip connection with e2
        x = self.upconv2(x)
        if x.size() != e2.size():
            x = nn.functional.interpolate(x, size=e2.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, e2], dim=1)  # Skip connection with e2
        x = self.dec2(x)
        
        # No skip connection for the last decoder block
        x = self.upconv1(x)
        x = self.dec1(x)
        
        # Make sure we get back to 120x120
        if x.size()[2:] != (120, 120):
            x = self.final_upconv(x)
        
        # Final convolution
        x = self.final_conv(x)
        return x
    
    def forward(self, x):
        # Encode
        encoded, skip_features = self.encoder(x)
        
        # Decode with skip connections
        decoded = self.decoder(encoded, skip_features)
        
        return decoded

class FaceAutoencoder120_skip(nn.Module):
    def __init__(self, bottleneck_size=1024):
        super(FaceAutoencoder120_skip, self).__init__()
        
        # Define encoder blocks individually to access intermediate outputs for skip connections
        # First encoder block
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 120x120 -> 60x60
        
        # Second encoder block
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 60x60 -> 30x30
        
        # Third encoder block
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 30x30 -> 15x15
        
        # Fourth encoder block
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 15x15 -> 7x7 (approx)
        
        # Flattened bottleneck
        self.flatten = nn.Flatten()
        
        # Calculate size after encoding convolutional layers
        # With 120x120 input, after 4 MaxPool layers (dividing by 2^4 = 16)
        # We get 7.5x7.5 which becomes 7x7
        self.feature_size = 256 * 7 * 7
        
        # Dense bottleneck layer - LARGER bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(self.feature_size, bottleneck_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder begins with dense layer back to pre-flatten size
        self.decoder_dense = nn.Sequential(
            nn.Linear(bottleneck_size, self.feature_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Define decoder blocks with transposed convolutions for better upsampling
        # First decoder block
        self.upconv4 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),  # 512 due to skip connection
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Second decoder block
        self.upconv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),  # 256 due to skip connection
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Third decoder block
        self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # 128 due to skip connection
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Fourth decoder block
        self.upconv1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),  # 64 due to skip connection
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Additional handling for odd dimensions
        self.final_upconv = nn.Upsample(size=(120, 120), mode='bilinear', align_corners=True)
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )

    def encoder(self, x):
        # Store intermediate outputs for skip connections
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        flat = self.flatten(p4)
        bottleneck = self.bottleneck(flat)
        
        # Store intermediate features for decoder
        return bottleneck, (e1, e2, e3, e4)
    
    def decoder(self, x, skip_features):
        e1, e2, e3, e4 = skip_features
        
        x = self.decoder_dense(x)
        x = x.view(-1, 256, 7, 7)  # Reshape to match encoder output dimensions
        
        # Use skip connections in decoder
        x = self.upconv4(x)
        # Handle potential size mismatch
        if x.size() != e4.size():
            x = nn.functional.interpolate(x, size=e4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, e4], dim=1)  # Skip connection
        x = self.dec4(x)
        
        x = self.upconv3(x)
        if x.size() != e3.size():
            x = nn.functional.interpolate(x, size=e3.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, e3], dim=1)  # Skip connection
        x = self.dec3(x)
        
        x = self.upconv2(x)
        if x.size() != e2.size():
            x = nn.functional.interpolate(x, size=e2.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, e2], dim=1)  # Skip connection
        x = self.dec2(x)
        
        x = self.upconv1(x)
        if x.size() != e1.size():
            x = nn.functional.interpolate(x, size=e1.size()[2:], mode='bilinear', align_corners=True)

           
        x = torch.cat([x, e1], dim=1)  # Skip connection
        x = self.dec1(x)
        
        # Make sure we get back to 120x120
        if x.size()[2:] != (120, 120):
            x = self.final_upconv(x)
        
        # Final convolution
        x = self.final_conv(x)
        return x
    
    def forward(self, x):
        # Encode
        encoded, skip_features = self.encoder(x)
        
        # Decode with skip connections
        decoded = self.decoder(encoded, skip_features)
        
        return decoded

class FaceDenoisingAutoencoder120(nn.Module):
    def __init__(self):
        super(FaceDenoisingAutoencoder120, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 120x120 -> 60x60
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 60x60 -> 30x30
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 30x30 -> 15x15
            
            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 15x15 -> 7x7 (approx)
        )
        
        # Flattened bottleneck
        self.flatten = nn.Flatten()
        
        # Calculate size after encoding convolutional layers
        # With 120x120 input, after 4 MaxPool layers (dividing by 2^4 = 16)
        # We get 7.5x7.5 which becomes 7x7
        self.bottleneck_size = 256 * 7 * 7
        
        # Dense bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Linear(self.bottleneck_size, 512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder begins with dense layer back to pre-flatten size
        self.decoder_dense = nn.Sequential(
            nn.Linear(512, self.bottleneck_size),
            nn.ReLU(inplace=True)
        )
        
        # Decoder convolutional layers
        self.decoder = nn.Sequential(
            # Reshape happens between decoder_dense and here
            
            # First upsampling block
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 7x7 -> 14x14
            
            # Second upsampling block
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 14x14 -> 28x28
            
            # Third upsampling block
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 28x28 -> 56x56
            
            # Fourth upsampling block
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 56x56 -> 112x112
            
            # Additional upsampling to get back to 120x120
            nn.Upsample(size=(120, 120), mode='bilinear', align_corners=True),
            
            # Final output layer
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )

    def encode(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.bottleneck(x)
        return x
    
    def decode(self, x):
        x = self.decoder_dense(x)
        x = x.view(-1, 256, 7, 7)  # Reshape to match encoder output dimensions
        x = self.decoder(x)
        return x
    def forward(self, x):
        # Encode
        encoded = self.encode(x)
        
        # Decode
        decoded = self.decode(encoded)
        
        return decoded
class FacesAutoencoder_112(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()

        # Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),   # -> 56x56
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), # -> 28x28
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),# -> 14x14
            nn.ReLU(True),
            #nn.Conv2d(256, 512, 4, 2, 1), # -> 7x7
            #nn.ReLU(True),
        )

        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, latent_dim)
        )

        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 14 * 14),
            nn.ReLU(True)
        )

        self.decoder_cnn = nn.Sequential(
            # nn.ConvTranspose2d(512, 256, 4, 2, 1) , # -> 14x14
            # nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # -> 28x28
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # -> 56x56
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),    # -> 112x112
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder_cnn(x)
        latent = self.encoder_fc(x)
        return latent

    def decode(self, latent):
        x = self.decoder_fc(latent)
        x = x.view(-1, 256, 14, 14)
        return self.decoder_cnn(x)

    def forward(self, x):
        latent = self.encode(x)
        out = self.decode(latent)
        return out



class FacesAutoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()

        # Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 112x112
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 56x56
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),  # 28x28
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1),  # 14x14
            nn.ReLU(True),
        )

        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 14 * 14, latent_dim)
        )

        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 14 * 14),
            nn.ReLU(True)
        )

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 28x28
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 56x56
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 112x112
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),  # 224x224
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder_cnn(x)
        latent = self.encoder_fc(x)
        return latent

    def decode(self, latent):
        x = self.decoder_fc(latent)
        x = x.view(-1, 512, 14, 14)
        return self.decoder_cnn(x)

    def forward(self, x):
        latent = self.encode(x)
        out = self.decode(latent)
        return out

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1600, 128)  # 1600 = 64 * 5 * 5
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 1600)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Contracting path
        self.enc1 = self.conv_block(1, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)

        # Bottom
        self.bottom = self.conv_block(64, 128)

        # Expanding path
        self.up3 = self.upconv_block(128, 64,output_padding=1)
        self.dec3 = self.conv_block(128, 64)

        self.up2 = self.upconv_block(64, 32)
        self.dec2 = self.conv_block(64, 32)

        self.up1 = self.upconv_block(32, 16)
        self.dec1 = self.conv_block(32, 16)

        # Final output layer
        self.final = nn.Conv2d(16, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels, output_padding=0):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2,output_padding=output_padding)



    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # (1, 64, 28, 28)
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))  # (64, 128, 14, 14)
        enc3 = self.enc3(nn.MaxPool2d(2)(enc2))  # (128, 256, 7, 7)

        # Bottom
        bottom = self.bottom(nn.MaxPool2d(2)(enc3))  # (256, 512, 3, 3)

        # Decoder
        up3 = self.up3(bottom)  # (512, 256, 7, 7)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))

        up2 = self.up2(dec3)  # (256, 128, 14, 14)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = self.up1(dec2)  # (128, 64, 28, 28)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        # Final layer
        out = self.final(dec1)  # (64, 1, 28, 28)
        return out


class MnistDnet(nn.Module):
    def __init__(self):
        super(MnistDnet, self).__init__()

        self.encode = nn.Sequential(

            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),

        )

    def forward(self, x):
        x = self.encode(x)
        return x

class MnistAutoencoder(nn.Module):
    def __init__(self, bottleneck_dim=32):
        super(MnistAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class MNISTConvDnet(nn.Module):
    def __init__(self):
        super(MNISTConvDnet, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # (1, 32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (32, 64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),  # Flatten to (64*7*7)
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        out=self.encoder(x)
        return out


class MNISTConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(MNISTConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # (1, 32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (32, 64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),  # Flatten to (64*7*7)
            nn.Linear(64 * 7 * 7, latent_dim)  # Latent space
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),  # Reshape back to spatial dimensions
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (64, 32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (32, 1, 28, 28)
            nn.Sigmoid()  # Normalize to [0, 1]
        )

    def freeze_decoder(self):
        # Freeze decoder weights
        for param in self.decoder.parameters():
            param.requires_grad = False

    def unfreeze_decoder(self):
        # Unfreeze decoder weights if needed later
        for param in self.decoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
    
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list = [64, 32, 16], dropout_rate: float = 0.2):
        """
        4-layer MLP with configurable input dimension and fixed output size of 1
        
        Args:
            input_dim: Number of input features
            hidden_sizes: List of hidden layer sizes (default: [64, 32, 16])
            dropout_rate: Dropout probability (default: 0.2)
        """
        super(MLP, self).__init__()
        
        # Layer definitions
        self.layer1 = nn.Linear(input_dim, hidden_sizes[0])
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.layer3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.layer4 = nn.Linear(hidden_sizes[2], 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using He initialization"""
        for layer in [self.layer1, self.layer2, self.layer3]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Layer 1
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Layer 3
        x = self.layer3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.layer4(x)
        return x    
    

class DiffusionModel(nn.Module):
    def __init__(self, latent_dim):
        super(DiffusionModel, self).__init__()

        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )

        # Simple MLP for the latent space
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 128, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1).float())

        # Concatenate and process through network
        x_t = torch.cat([x, t_emb], dim=1)
        return self.net(x_t)   
    

class CifarAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(CifarAutoencoder, self).__init__()
       # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 16x16x32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 8x8x64
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 4x4x128
        )
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Fully connected layers for latent space
        self.fc_encoder = nn.Sequential(
            nn.Linear(4 * 4 * 128, 256),
            nn.ReLU(True),
            nn.Linear(256, latent_dim),
            nn.ReLU(True)
        )
        
        # Decoder fully connected layers
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 4 * 4 * 128),
            nn.ReLU(True)
        )
        
        # Decoder convolutional layers
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),  # 8x8x128
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),  # 16x16x64
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),  # 32x32x32
            
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Output values between 0 and 1
        )
    
    def encode(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc_encoder(x)
        return x
    
    def decode(self, z):
        z = self.fc_decoder(z)
        z = self.decoder(z)
        return z
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon  
    

class CifarAutoencoder2(nn.Module):
    def __init__(self, latent_dim=512):
        super(CifarAutoencoder2, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (128, 8, 8)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (256, 4, 4)
            nn.ReLU(),
            nn.Flatten(),  # (256*4*4 = 4096)
            nn.Linear(4096, latent_dim),  # Latent space
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4096),  # Expand back to 4096
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),  # Reshape back to (256, 4, 4)
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (128, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),  # (3, 32, 32)
            nn.Sigmoid()  # Normalize to [0,1]
        )

    def forward(self, x):
        x = self.encoder(x)  # Encode to latent vector
        x = self.decoder(x)  # Decode back to image
        return x
