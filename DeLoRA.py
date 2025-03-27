import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DeLoRALinear(nn.Module):
    """
    DeLoRALinear implements the Decoupled Low-Rank Adaptation (DeLoRA) as described in:
    "Decoupling Angles and Strength in Low-rank Adaptation" (Bini et al., ICLR 2025).

    Given a frozen pretrained weight W ∈ ℝ^(out_features x in_features),
    DeLoRA applies a low-rank update via:
    
        W' = W + (λ * ||W||_F / r) * (B_norm @ A_norm)
    
    where:
      • A ∈ ℝ^(r x in_features) and B ∈ ℝ^(out_features x r) are learnable,
      • B_norm and A_norm are computed by normalizing each rank-1 component (columns of B, rows of A),
      • λ is a learnable scaling parameter, and
      • r is the low-rank dimension.
    
    This formulation decouples the “angular” (directional) component from the update strength.
    The module also provides merge/unmerge functions for efficient inference.
    """
    def __init__(self, in_features: int, out_features: int, r: int = 4, lambda_init: float = 1e-3, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r

        # Base (pretrained) weight – set to be frozen.
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight.requires_grad = False

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Low-rank adaptation parameters.
        # A: (r, in_features), B: (out_features, r)
        self.A = nn.Parameter(torch.Tensor(r, in_features))
        self.B = nn.Parameter(torch.Tensor(out_features, r))
        
        # Learnable scaling parameter controlling the update strength.
        self.lambda_ = nn.Parameter(torch.tensor(lambda_init, dtype=torch.float32))
        
        self.reset_parameters()
        # Save a backup of the original (pretrained) weight for merge/unmerge operations.
        self.register_buffer("weight_orig", self.weight.data.clone())
        self.merged = False

    def reset_parameters(self):
        # Initialize the base weight and bias in a standard way.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        # Initialize the low-rank matrices.
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        # Optionally, zero-init the low-rank updates so that the module starts as the base model.
        nn.init.zeros_(self.A)
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            # If merged, simply use the adapted weight.
            return F.linear(x, self.weight, self.bias)
        else:
            eps = 1e-6
            # Normalize each column of B and each row of A to decouple the angular component.
            B_norm = self.B / (self.B.norm(dim=0, keepdim=True) + eps)  # shape: (out_features, r)
            A_norm = self.A / (self.A.norm(dim=1, keepdim=True) + eps)    # shape: (r, in_features)
            # Compute the low-rank update.
            update = (self.lambda_ * self.weight.norm(p='fro') / self.r) * (B_norm @ A_norm)
            # Use base weight + update.
            W_adapted = self.weight + update
            return F.linear(x, W_adapted, self.bias)

    def merge_weights(self):
        """
        Merges the low-rank update into the base weight for efficient inference.
        Stores the original weight in a buffer and sets a flag.
        """
        if not self.merged:
            eps = 1e-6
            B_norm = self.B / (self.B.norm(dim=0, keepdim=True) + eps)
            A_norm = self.A / (self.A.norm(dim=1, keepdim=True) + eps)
            update = (self.lambda_ * self.weight.norm(p='fro') / self.r) * (B_norm @ A_norm)
            # Merge update into the base weight.
            self.weight.data = self.weight_orig + update.data
            self.merged = True

    def unmerge_weights(self):
        """
        Reverts the merged weight back to the original (pretrained) weight.
        """
        if self.merged:
            self.weight.data = self.weight_orig.clone()
            self.merged = False

# 예시 사용법
if __name__ == '__main__':
    batch_size, in_features, out_features = 8, 128, 64
    x = torch.randn(batch_size, in_features)
    
    # DeLoRA Linear layer with low-rank update of rank 4.
    de_lora = DeLoRALinear(in_features, out_features, r=4, lambda_init=1e-3)
    
    # Forward pass without merging (training mode)
    y = de_lora(x)
    print("Output shape (unmerged):", y.shape)
    
    # Merge the weights for efficient inference.
    de_lora.merge_weights()
    y_merged = de_lora(x)
    print("Output shape (merged):", y_merged.shape)
    
    # Unmerge to revert to original state.
    de_lora.unmerge_weights()
    y_unmerged = de_lora(x)
    print("Output shape (unmerged again):", y_unmerged.shape)
