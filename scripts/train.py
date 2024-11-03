import torch
from torch.utils.data import DataLoader
import yaml
from data.data_preprocessing import prepare_data 
from models.pix2pix_model import Pix2PixModel
from utils.losses import Pix2PixLosses 

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Load configuration
    config = load_config()

    # Prepare data
    train_dataset, val_dataset = prepare_data(config['data']['train_data_dir'])
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # Initialize model and losses
    model = Pix2PixModel(config)
    model.setup(config)
    losses = Pix2PixLosses(lambda_L1=config['training']['lambda_L1'], gan_mode='vanilla')  # Initialize losses with lambda_L1

    # Set up device for training
    device = torch.device("cuda" if config['device']['use_gpu'] and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    for epoch in range(config['training']['start_epoch'], config['training']['epochs'] + 1):
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        model.train()
        
        for i, data in enumerate(train_loader):
            # Forward pass and generate images
            model.set_input(data)
            model.optimize_parameters()  # Updates model's generator and discriminator

            # Calculate losses
            pred_fake = model.netD(model.fake_B)
            pred_real = model.netD(model.real_B)
            loss_G = losses.generator_loss(pred_fake, model.real_B, model.fake_B)
            loss_D = losses.discriminator_loss(pred_real, pred_fake)

            # Print or log loss values
            if i % config['logging']['log_interval'] == 0:
                print(f"Step [{i}/{len(train_loader)}]: Generator Loss - {loss_G.item()}, Discriminator Loss - {loss_D.item()}")

        # Save model checkpoint
        if epoch % config['logging']['checkpoint_interval'] == 0:
            print(f"Saving model at epoch {epoch}")
            model.save_networks(epoch)
    
    # Final checkpoint
    model.save_networks('latest')
    print("Training complete.")

if __name__ == "__main__":
    main()
