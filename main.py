import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('Model'), "..")))

import torch
import timm
from Model.Encoder import Encoder
from Model.TransModule import TransModule, TransModule_Config
from Model.Decoder import Decoder
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image

# Load Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(
    img_size=224,
    patch_size=2,
    in_chans=3,
    embed_dim=192,
    depths=[2, 2, 2],
    nhead=[3, 6, 12],
    strip_width=[2, 4, 7],
    drop_path_rate=0.,
    patch_norm=True
).to(device)
trans_config = TransModule_Config(nlayer=3, d_model=768, nhead=8, norm_first=True)
transfer_module = TransModule(config=trans_config).to(device)
decoder = Decoder(d_model=768, seq_input=True).to(device)

checkpoint = torch.load('../.models/model_complete/checkpoint_40000_epoch.pkl')

# Load the state dictionaries
encoder.load_state_dict(checkpoint['encoder'])
transfer_module.load_state_dict(checkpoint['transModule'])
decoder.load_state_dict(checkpoint['decoder'])

encoder.eval()
transfer_module.eval()
decoder.eval()

# Paths to images
content_img_path = './image_content.png'
style_img_path = './image_style.png'

# Load images
content_img = Image.open(content_img_path).convert('RGB')
style_img = Image.open(style_img_path).convert('RGB')

content_shape = content_img.size
size = min(content_shape[1], content_shape[0])

# Preprocess the images
only_tensor_transforms = T.Compose([
    T.ToTensor(),
])

shape_transform = T.Compose([
    T.Resize(size),
    T.ToTensor(),
])

content_img = only_tensor_transforms(content_img).unsqueeze(0).to(device)
style_img = shape_transform(style_img).unsqueeze(0).to(device)

# Forward pass through encoders
forward_content = encoder(content_img, arbitrary_input = True)  # [b, h, w, c]
forward_style = encoder(style_img, arbitrary_input = True)      # [b, h, w, c]

content_features, content_res = forward_content[0], forward_content[2]  # [b, c, h, w]
style_features, style_res = forward_style[0], forward_style[2]      # [b, c, h, w]

# Merge the features
merged_features = transfer_module(content_features, style_features)

# Decode the merged features
output = decoder(merged_features, content_res)  # [b, c, h, w]

# Save the generated image
save_image(output, 'generated_image.png')