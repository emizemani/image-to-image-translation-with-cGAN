import sys
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import UNetGenerator
from utils.helper_functions import load_config
from torchvision import transforms


def analyze_generator(config, input_dir, output_dir, baseline=None, steps=50):
    # load generator
    generator = UNetGenerator()
    
    # Use the same paths as in train_apply.py
    best_generator_path = os.path.join(config['logging']['checkpoint_dir'], 'best_model/generator_latest.pth')
    
    device = torch.device("cuda" if config['device']['use_gpu'] and torch.cuda.is_available() else "cpu")

    # Load with appropriate device mapping
    generator.load_state_dict(
        torch.load(best_generator_path, device, weights_only=True)
    )
    generator = generator.to(device)
    generator.eval()

    image = Image.open(input_dir).convert("RGB")
    transform = transforms.ToTensor()
    image = transform(image)

    input_image = image.to(device)

    # attributions = integrated_gradients(generator, input_image, baseline, steps)
    # visualize_integrated_gradients(attributions, input_image)

    target_layer = generator.encoder[2]

    heatmap = grad_cam(generator, target_layer, input_image)

    # Overlay heatmap on the image
    overlay = overlay_heatmap_on_image(heatmap, input_image)


def integrated_gradients(model, input_image, baseline, steps):

    if baseline is None:
        baseline = torch.zeros_like(input_image)  # Default to black baseline

    # Scale inputs from baseline to the actual input
    scaled_inputs = [
        baseline + (float(i) / steps) * (input_image - baseline) 
        for i in range(steps + 1)
    ]
    scaled_inputs = torch.stack(scaled_inputs, dim=0)

    # Compute gradients at each step
    scaled_inputs.requires_grad = True
    model.zero_grad()

    # Forward pass and backward pass to compute gradients
    output = model(scaled_inputs)
    loss = F.cross_entropy(output, torch.argmax(output, dim=1))  # Example for classification
    loss.backward(torch.ones_like(loss), retain_graph=True)

    # Compute gradients and average over steps
    gradients = scaled_inputs.grad
    avg_gradients = gradients.mean(dim=0)

    # Compute attribution: (input - baseline) * avg_gradients
    attributions = (input_image - baseline) * avg_gradients
    return attributions

def visualize_integrated_gradients(attributions, input_image):
    """
    Visualizes the Integrated Gradients attributions on the input image.
    
    Args:
        attributions: The attribution scores (same shape as input image).
        input_image: The original input image (tensor of shape (C, H, W)).
    """
    # Convert the input image to numpy for visualization
    input_image = input_image.squeeze().cpu().detach().numpy()
    input_image = np.transpose(input_image, (1, 2, 0))  # CxHxW -> HxWxC
    input_image = np.uint8(255 * input_image)

    # Convert attributions to numpy
    attributions = attributions.squeeze().cpu().detach().numpy()
    attributions = np.abs(attributions)  # Take the absolute value of attributions
    attributions = attributions.sum(axis=0)  # Sum over channels for RGB images
    attributions = cv2.resize(attributions, (input_image.shape[1], input_image.shape[0]))
    
    # Normalize the attributions for visualization
    attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min())

    # Apply a colormap to the attributions
    heatmap = np.uint8(255 * attributions)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on the image
    superimposed_image = cv2.addWeighted(input_image, 0.5, heatmap, 0.5, 0)
    
    # Plot the image and the heatmap
    plt.imshow(superimposed_image)
    plt.axis('off')
    plt.show()


def grad_cam(model, target_layer, input_image, target_class=None):
    """
    Implements Grad-CAM for a PyTorch model.
    
    Args:
        model: The PyTorch model to interpret.
        target_layer: The name of the layer where Grad-CAM will focus.
        input_image: The input tensor (1, C, H, W) for which Grad-CAM is computed.
        target_class: The target class index. If None, uses the class with the highest prediction score.

    Returns:
        heatmap: The generated Grad-CAM heatmap (H, W).
    """
    gradients = []
    activations = []
    
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])  # Store gradients from backward pass

    def forward_hook(module, input, output):
        activations.append(output)  # Store feature maps from forward pass

    # Register hooks
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)
    
    # Forward pass
    model.eval()
    input_image = input_image.unsqueeze(0)
    output = model(input_image)
    
    # Determine the target class
    if target_class is None:
        target_class = output.argmax(dim=1)

    # Backward pass
    model.zero_grad()
    target = output[:, target_class]
    target.backward()

    # Get stored activations and gradients
    activations = activations[0].detach()
    gradients = gradients[0].detach()

    # Compute the weights
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # Global average pooling

    # Compute Grad-CAM
    grad_cam_map = (weights * activations).sum(dim=1, keepdim=True)
    grad_cam_map = F.relu(grad_cam_map)  # ReLU to keep positive importance
    grad_cam_map = grad_cam_map.squeeze()  # Remove unnecessary dimensions

    # Normalize the Grad-CAM heatmap
    grad_cam_map -= grad_cam_map.min()
    grad_cam_map /= grad_cam_map.max()
    
    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()

    return grad_cam_map.cpu().numpy()  # Convert to NumPy array for visualization

def overlay_heatmap_on_image(heatmap, original_image, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlays a heatmap onto the original image.
    
    Args:
        heatmap (numpy.ndarray): The Grad-CAM heatmap, normalized to range [0, 1].
        original_image (PIL.Image): The original image.
        alpha (float): Opacity factor for the heatmap overlay (0 = invisible, 1 = opaque).
        colormap: OpenCV colormap to colorize the heatmap (e.g., cv2.COLORMAP_JET).
    
    Returns:
        overlay (numpy.ndarray): The heatmap overlaid on the original image.
    """
    # Convert the original image to a NumPy array
    original_image = np.array(original_image)
    
    # Normalize heatmap to range [0, 255] and apply colormap
    heatmap = np.uint8(255 * heatmap)  # Scale to 0-255
    heatmap_color = cv2.applyColorMap(heatmap, colormap)  # Apply colormap
    
    # Resize heatmap to match the original image size
    heatmap_color = cv2.resize(heatmap_color, (original_image.shape[1], original_image.shape[0]))
    
    # Convert the original image to BGR format if it's RGB
    if original_image.shape[-1] == 3 and original_image.dtype == np.uint8:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    
    # Blend heatmap with the original image
    overlay = cv2.addWeighted(heatmap_color, alpha, original_image, 1 - alpha, 0)
    
    # Convert back to RGB for display
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay

def visualize_grad_cam(overlay):
    plt.imshow(overlay)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":

    # Load configuration
    config_path = "config.yaml"
    config = load_config(config_path)

    # Define input directory
    input_dir = "validation/test4/0/image_025.png"

    # Define output directory
    output_dir = "explain/test2"

    analyze_generator(config, input_dir, output_dir)



