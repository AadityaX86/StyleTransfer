import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import Normalize

# Load a pretrained VGG-19 model and modify it to return feature maps
class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG19FeatureExtractor, self).__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.vgg_layers = vgg19[:30]
        self.target_layers = {1, 6, 11, 20, 29}  # relu1_1, relu2_1, ..., relu5_1

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if i in self.target_layers:
                features.append(x)
        return features


# Compute channel-wise means and variances of features
def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert len(size) == 4, 'The shape of feature needs to be a tuple with length 4.'
    B, C = size[:2]
    feat_mean = feat.reshape(B, C, -1).mean(dim=2).reshape(B, C, 1, 1)
    feat_std = (feat.reshape(B, C, -1).var(dim=2, unbiased=False) + eps).sqrt().reshape(B, C, 1, 1)
    return feat_mean, feat_std


# Normalize features
def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


# Content and Style Loss
class ContentStyleLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=3.0, sigma = 50.0, gamma = 1.0):
        """
        Initialize the loss function.
        :param alpha: Weight for content loss
        :param beta: Weight for style loss
        """
        super(ContentStyleLoss, self).__init__()
        self.vgg = VGG19FeatureExtractor()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha  # Weight for content loss
        self.beta = beta    # Weight for style loss
        self.sigma = sigma # Weight for identity_loss_1
        self.gamma = gamma # Weight for identity_loss_2

    def forward(self, output, identity_image_content, identity_image_style, content_image, style_image):
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        content_image = normalize(content_image)
        style_image = normalize(style_image)
        identity_image_content = normalize(identity_image_content)
        identity_image_style = normalize(identity_image_style)
        output = normalize(output)

        # Extract feature maps
        out_1, out_2, out_3, out_4, out_5 = self.vgg(output)
        content_1, content_2, content_3, content_4, content_5 = self.vgg(content_image)
        style_1, style_2, style_3, style_4, style_5 = self.vgg(style_image)
        identity_image_content_1, identity_image_content_2, identity_image_content_3, identity_image_content_4, identity_image_content_5 = self.vgg(identity_image_content)
        identity_image_style_1, identity_image_style_2, identity_image_style_3, identity_image_style_4, identity_image_style_5 = self.vgg(identity_image_style)

        # Content loss (MSE between normalized feature maps at relu3_1 and relu4_1)
        content_loss = self.mse_loss(mean_variance_norm(out_4), mean_variance_norm(content_4)) + self.mse_loss(mean_variance_norm(out_5), mean_variance_norm(content_5))

        # Style loss (MSE between mean and std of feature maps)
        style_loss = 0
        for out_feature, style_feature in zip([out_2, out_3, out_4, out_5], [style_2, style_3, style_4, style_5]):
            out_mean, out_std = calc_mean_std(out_feature)
            style_mean, style_std = calc_mean_std(style_feature)
            style_loss += self.mse_loss(out_mean, style_mean) + self.mse_loss(out_std, style_std)

        identity_loss_1 = self.mse_loss(identity_image_content, content_image) + self.mse_loss(identity_image_style, style_image)

        identity_loss_2 = 0
        for content_feature, style_feature, identity_image_content_feature, identity_image_style_feature in zip([content_2, content_3, content_4, content_5], 
                                                                                                                [style_2, style_3, style_4, style_5], 
                                                                                                                [identity_image_content_2, identity_image_content_3, identity_image_content_4, identity_image_content_5], 
                                                                                                                [identity_image_style_2, identity_image_style_3, identity_image_style_4, identity_image_style_5]):
            identity_loss_2 += self.mse_loss(content_feature, identity_image_content_feature) + self.mse_loss(style_feature, identity_image_style_feature)

        # Total loss: Weighted sum of content and style loss
        total_loss = self.alpha * content_loss + self.beta * style_loss + self.sigma * identity_loss_1 + self.gamma * identity_loss_2

        return total_loss, content_loss, style_loss, identity_loss_1, identity_loss_2
    
if __name__ == "__main__":
    
    # Dummy output and target images (B, 3, 256, 256)
    output = torch.rand(2, 3, 256, 256)  # Values in [0,1]
    content_target = torch.rand(2, 3, 256, 256)
    style_target = torch.rand(2, 3, 256, 256)

    # Initialize the loss function with weights for content and style
    loss_fn = ContentStyleLoss()

    # Compute the total loss, content loss, and style loss
    total_loss, content_loss, style_loss, identity_loss_1, identity_loss_2 = loss_fn(output, content_target, style_target)

    print(f"Total Loss: {total_loss.item()}")
    print(f"Content Loss: {content_loss.item()}")
    print(f"Style Loss: {style_loss.item()}")
    print(f"Identity_Loss_1: {identity_loss_1.item()}")
    print(f"Identity_Loss_2: {identity_loss_2.item()}")