import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.helper_functions import load_config
from data.dataset import CustomDataset
from src.model import UNetGenerator, PatchGANDiscriminator
from utils.losses import GANLosses


def train_model(config):
    """
    Train the generator and discriminator using the specified configuration.
    """
    # Prepare datasets
    train_dataset = CustomDataset(
        images_dir=config['data']['train_images_dir'],
        labels_dir=config['data']['train_labels_dir'],
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
    )

    val_dataset = CustomDataset(
        images_dir=config['data']['val_images_dir'],
        labels_dir=config['data']['val_labels_dir'],
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
    )

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # Initialize generator and discriminator
    generator = UNetGenerator()
    discriminator = PatchGANDiscriminator()

    # Set up device for training
    device = torch.device("cuda" if config['device']['use_gpu'] and torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    # Initialize optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config['training']['lr'], betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config['training']['lr'], betas=(0.5, 0.999))

    # Initialize loss functions
    losses = GANLosses(lambda_L1=config['training']['lambda_L1'], gan_mode='vanilla')

    # Training loop
    for epoch in range(config['training']['start_epoch'], config['training']['epochs'] + 1):
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        generator.train()
        discriminator.train()

        for i, data in enumerate(train_loader):
            # Unpack and move data to device
            real_A, real_B = data['A'].to(device), data['B'].to(device)

            # -----------------------------
            # Train Generator
            # -----------------------------
            optimizer_G.zero_grad()
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            loss_G = losses.generator_loss(pred_fake, real_B, fake_B)
            loss_G.backward()
            optimizer_G.step()

            # --------------------------------
            # Train Discriminator
            # --------------------------------
            optimizer_D.zero_grad()
            pred_real = discriminator(real_B, real_A)
            pred_fake_detached = discriminator(fake_B.detach(), real_A)
            loss_D = losses.discriminator_loss(pred_real, pred_fake_detached)
            loss_D.backward()
            optimizer_D.step()

            # Log losses at specified intervals
            if i % config['logging']['log_interval'] == 0:
                print(f"Step [{i}/{len(train_loader)}]: Generator Loss - {loss_G.item():.4f}, Discriminator Loss - {loss_D.item():.4f}")

        # Save model checkpoints at specified intervals
        if epoch % config['logging']['checkpoint_interval'] == 0:
            print(f"Saving model at epoch {epoch}")
            torch.save(generator.state_dict(), f"{config['logging']['checkpoint_dir']}/generator_epoch_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"{config['logging']['checkpoint_dir']}/discriminator_epoch_{epoch}.pth")

        # Validation phase
        if epoch % config['logging']['validation_interval'] == 0:
            validate_model(generator, val_loader, device, losses)

    # Save the final model
    print("Saving final model...")
    torch.save(generator.state_dict(), f"{config['logging']['checkpoint_dir']}/generator_latest.pth")
    torch.save(discriminator.state_dict(), f"{config['logging']['checkpoint_dir']}/discriminator_latest.pth")
    print("Training complete.")


def validate_model(generator, val_loader, device, losses):
    """
    Perform validation on the generator.
    Args:
        generator (nn.Module): Trained generator model.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Device for computation.
        losses (GANLosses): Loss functions for evaluation.
    """
    generator.eval()
    total_loss = 0
    with torch.no_grad():
        for data in val_loader:
            real_A, real_B = data['A'].to(device), data['B'].to(device)
            fake_B = generator(real_A)
            loss = losses.l1_loss(fake_B, real_B).item()
            total_loss += loss

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    config = load_config()
    train_model(config)
