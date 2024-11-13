import sys
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
from data.dataset import CustomDataset  # Ensure this path is correct
from model import UNetGenerator, PatchGANDiscriminator
from utils.losses import GANLosses

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def main():

    config = load_config()

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
    #for later val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # initialize generator and discriminator
    generator = UNetGenerator()
    discriminator = PatchGANDiscriminator()

    #set up device for training
    device = torch.device("cuda" if config['device']['use_gpu'] and torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    #initialize optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config['training']['lr'], betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config['training']['lr'], betas=(0.5, 0.999))

    #initialize loss functions
    losses = GANLosses(lambda_L1=config['training']['lambda_L1'], gan_mode='vanilla')

    #training loop
    for epoch in range(config['training']['start_epoch'], config['training']['epochs'] + 1):
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        generator.train()
        discriminator.train()

        for i, data in enumerate(train_loader):
            # unpack and move data to device
            real_A, real_B = data['A'].to(device), data['B'].to(device)

            # for debbuging, can be removed later
            print(f"\nBatch {i}:")
            print(f"Shape of real_A: {real_A.shape}")
            print(f"Shape of real_B: {real_B.shape}")

            # -----------------------------
            # Train Generator
            # -----------------------------
            optimizer_G.zero_grad()

            # generate fake image
            fake_B = generator(real_A)

            # debbuging
            print(f"Shape of fake_B: {fake_B.shape}")

            # GAN loss for the generator
            pred_fake = discriminator(fake_B, real_A)
            loss_G = losses.generator_loss(pred_fake, real_B, fake_B)

            loss_G.backward()
            optimizer_G.step()

            # --------------------------------
            # Train Discriminator
            # --------------------------------
            optimizer_D.zero_grad()

            #real loss
            pred_real = discriminator(real_B, real_A)
            pred_fake_detached = discriminator(fake_B.detach(), real_A)

            loss_D = losses.discriminator_loss(pred_real, pred_fake_detached)
            loss_D.backward()
            optimizer_D.step()

            if i % config['logging']['log_interval'] == 0:
                print(f"Step [{i}/{len(train_loader)}]: Generator Loss - {loss_G.item()}, Discriminator Loss - {loss_D.item()}")

        #save model checkpoints
        if epoch % config['logging']['checkpoint_interval'] == 0:
            print(f"Saving model at epoch {epoch}")
            torch.save(generator.state_dict(), f"{config['logging']['checkpoint_dir']}/generator_epoch_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"{config['logging']['checkpoint_dir']}/discriminator_epoch_{epoch}.pth")

    # final checkpoint
    torch.save(generator.state_dict(), f"{config['logging']['checkpoint_dir']}/generator_latest.pth")
    torch.save(discriminator.state_dict(), f"{config['logging']['checkpoint_dir']}/discriminator_latest.pth")
    print("Training complete.")

if __name__ == "__main__":
    main()
