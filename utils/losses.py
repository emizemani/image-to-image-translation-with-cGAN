import torch
import torch.nn as nn

class ConditionalGANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla'):
        super(ConditionalGANLoss, self).__init__()
        if gan_mode == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits
        elif gan_mode == 'lsgan':
            self.criterion = nn.MSELoss()  # LSGAN uses mean squared error
        else:
            raise NotImplementedError(f"GAN mode '{gan_mode}' not implemented")

    def forward(self, prediction, target_is_real):
        # Ensure `prediction` is a tensor
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("Prediction must be a tensor")

        # Create a target tensor based on `target_is_real`
        target_tensor = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)

        # Compute the loss
        loss = self.criterion(prediction, target_tensor)
        return loss

class GANLosses:
    def __init__(self, lambda_L1=100.0, gan_mode='vanilla'):
        self.lambda_L1 = lambda_L1
        self.gan_loss = ConditionalGANLoss(gan_mode=gan_mode)
        self.l1_loss = nn.L1Loss()

    def generator_loss(self, pred_fake, real_image, fake_image):
        """Calculate the combined loss for the generator."""
        # GAN loss - making discriminator think the fake image is real
        loss_G_GAN = self.gan_loss(pred_fake, target_is_real=True)
        
        # L1 loss - pixel-wise similarity between fake and real images
        loss_L1 = self.l1_loss(fake_image, real_image) * self.lambda_L1
        
        # Total generator loss
        loss_G = loss_G_GAN + loss_L1
        return loss_G

    def discriminator_loss(self, pred_real, pred_fake):
        """Calculate the loss for the discriminator."""
        # Discriminator loss for real images (should classify them as real)
        loss_D_real = self.gan_loss(pred_real, target_is_real=True)
        
        # Discriminator loss for fake images (should classify them as fake)
        loss_D_fake = self.gan_loss(pred_fake, target_is_real=False)
        
        # Total discriminator loss (average of real and fake)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D
