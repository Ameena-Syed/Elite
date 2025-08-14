import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image transform
imsize = 400
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load image
def load_image(path):
    image = Image.open(path).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device)

# Convert tensor to image
def im_convert(tensor):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    image = image.clamp(0, 1)
    return image.permute(1, 2, 0).numpy()

# Load your images
content = load_image("content.jpg")
style = load_image("style.jpg") 
target = content.clone().requires_grad_(True)

# Show input images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(im_convert(content))
ax1.set_title("Content Image")
ax1.axis("off")
ax2.imshow(im_convert(style))
ax2.set_title("Style Image")
ax2.axis("off")
plt.show()

# Load pre-trained VGG
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

# Freeze parameters
for param in vgg.parameters():
    param.requires_grad = False

# Map layer numbers to layer names
def get_features(image, model):
    features = {}
    x = image
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',  # content layer
        '28': 'conv5_1'
    }
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# Gram matrix
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())

# Get features
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)
target_features = get_features(target, vgg)

# Compute gram matrices for style image
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Layer weights
style_weights = {
    'conv1_1': 1.0,
    'conv2_1': 0.8,
    'conv3_1': 0.5,
    'conv4_1': 0.3,
    'conv5_1': 0.1
}
content_weight = 1e4
style_weight = 1e2

# Optimizer
optimizer = optim.Adam([target], lr=0.003)

# Training loop
epochs = 500
for i in range(epochs):
    target_features = get_features(target, vgg)
    
    # Content loss
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
    
    # Style loss
    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
        style_loss += layer_style_loss
    
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if i % 100 == 0:
        print(f"Epoch {i}, Total loss: {total_loss.item():.2f}")

# Show final output
plt.figure(figsize=(6, 6))
plt.imshow(im_convert(target))
plt.title("Output Image")
plt.axis('off')
plt.show()

# Save result
plt.imsave("output.jpg", im_convert(target))
print("✅ Final image saved as 'output.jpg'")