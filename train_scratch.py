import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from Model.Encoder import Encoder
from Model.TransModule import TransModule, TransModule_Config
from Model.Decoder import Decoder
from Model.Loss import ContentStyleLoss
from Sampler import SimpleDataset, InfiniteSamplerWrapper
from Scheduler import CosineAnnealingWarmUpLR
from NetworkLogger import NetworkLogger

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2
num_iterations = 80000  # Total number of training iterations
learning_rate = 0.5e-4

dataset_content = SimpleDataset('./Data/train/content_224', transforms=T.ToTensor())
dataset_style = SimpleDataset('./Data/train/style_224', transforms=T.ToTensor())
sampler_content = InfiniteSamplerWrapper(dataset_content)
sampler_style = InfiniteSamplerWrapper(dataset_style)
dataloader_content_iter = iter(DataLoader(dataset_content, batch_size=batch_size, sampler=sampler_content, num_workers=0))
dataloader_style_iter = iter(DataLoader(dataset_style, batch_size=batch_size, sampler=sampler_style, num_workers=0))

# Initialize TensorBoard logger
logger = NetworkLogger("./.logs/logs_scratch/")

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
loss_fn = ContentStyleLoss().to(device)

# Optimizer and learning rate scheduler
optimizer = optim.Adam([
    {'params': encoder.parameters()},
    {'params': decoder.parameters()},
    {'params': transfer_module.parameters()},
], lr = learning_rate)
scheduler = CosineAnnealingWarmUpLR(optimizer, warmup_step=80000//4, max_step=80000, min_lr=0)

# Set the models to training mode
encoder.train()
transfer_module.train()
decoder.train()

total_loss_accum = 0
content_loss_accum = 0
style_loss_accum = 0
identity_loss_1_accum = 0
identity_loss_2_accum = 0

for iteration in range(1, num_iterations + 1):
    # Load content and style images
    content_images = next(dataloader_content_iter).to(device)
    style_images = next(dataloader_style_iter).to(device)

    # Forward pass through encoders
    forward_content = encoder(content_images)  # [b, h, w, c]
    forward_style = encoder(style_images)      # [b, h, w, c]

    content_features, content_res = forward_content[0], forward_content[2]  # [b, c, h, w]
    style_features, style_res = forward_style[0], forward_style[2]      # [b, c, h, w]

    # Merge the features
    merged_features = transfer_module(content_features, style_features)

    # Decode the merged features
    output = decoder(merged_features, content_res)  # [b, c, h, w]

    identity_features_content = transfer_module(content_features, content_features)
    identity_features_style = transfer_module(style_features, style_features)

    identity_image_content = decoder(identity_features_content, content_res)
    identity_image_style = decoder(identity_features_style, style_res)

    total_loss, content_loss, style_loss, identity_loss_1, identity_loss_2 = loss_fn(output, identity_image_content, identity_image_style, content_images, style_images)
    
    total_loss_accum += total_loss.item()
    content_loss_accum += content_loss.item()
    style_loss_accum += style_loss.item()
    identity_loss_1_accum += identity_loss_1.item()
    identity_loss_2_accum += identity_loss_2.item()
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    scheduler.step()

    if iteration % 10 == 0:
        avg_total_loss = total_loss_accum / 10
        avg_content_loss = content_loss_accum / 10
        avg_style_loss = style_loss_accum / 10
        avg_identity_loss_1 = identity_loss_1_accum / 10
        avg_identity_loss_2 = identity_loss_2_accum / 10
        
        logger.log_scalars("Losses",{
            "Avg Total Loss": avg_total_loss,
            "Avg Content Loss": avg_content_loss,
            "Avg Style Loss": avg_style_loss,
            "Identity Loss 1": avg_identity_loss_1,
            "Identity Loss 2": avg_identity_loss_2
        }, iteration)
        print(f"Iteration {iteration}: Total Loss: {avg_total_loss}, Content Loss: {avg_content_loss}, Style Loss: {avg_style_loss}, IL_1: {identity_loss_1.item()}, IL_2: {identity_loss_2.item()}")
        
        total_loss_accum = 0
        content_loss_accum = 0
        style_loss_accum = 0
        identity_loss_1_accum = 0
        identity_loss_2_accum = 0

    if iteration % 1000 == 0:
        torch.save({
            'encoder': encoder.state_dict(),
            'transfer_module': transfer_module.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'iteration': iteration
        }, f"./.models/models_scratch/model_iter_{iteration}.pth")
        print(f"Checkpoint saved at iteration {iteration}")

print("Training complete!")
