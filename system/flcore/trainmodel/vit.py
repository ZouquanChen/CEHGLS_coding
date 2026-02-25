import math
import torch
from torch import nn, Tensor


class VisionTransformerModel(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 768,
        nhead: int = 1,
        nlayers: int = 1,
        num_classes: int = 1000,
        dropout: float = 0.5,
        d_hid: int = 3072,
    ):

        super().__init__()
        self.model_type = "VisionTransformer"

        # Patch embedding
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )

        # Learnable parameters
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        self.dropout = nn.Dropout(p=dropout)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            d_hid,
            dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN structure for ViT
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, nlayers)

        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

        self.init_weights()

    def init_weights(self) -> None:
        # Initialize cls token and position encoding
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

        # Linear projection layer initialization
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.normal_(self.patch_embed.bias, std=1e-6)

        # Classification head initialization
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, channels, height, width]

        Returns:
            output Tensor of shape [batch_size, num_classes]
        """
        # 1. Patch embedding
        x = self.patch_embed(x)  # [B, C, H, W] -> [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]

        # 2. Add cls token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [B, N+1, D]

        # 3. Add positional encoding
        x = x + self.pos_embed
        x = self.dropout(x)

        # 4. Pass through Transformer encoder
        x = self.encoder(x)

        # 5. Extract cls token and classify
        x = self.norm(x[:, 0])  # Take output from first position
        output = self.fc(x)

        return output
