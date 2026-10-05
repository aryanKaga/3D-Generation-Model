import torch
import torch.nn as nn

embd_dims = 768  # or 728 if that's really your feature size

class SAttention(nn.Module):
    def __init__(self, embd_dims=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.embd_dims = embd_dims
        self.num_heads = num_heads
        self.dk = embd_dims // num_heads

        self.cls_token = nn.Parameter(torch.randn(1, 1, embd_dims))
        self.q_proj = nn.Linear(embd_dims, embd_dims)
        self.k_proj = nn.Linear(embd_dims, embd_dims)
        self.v_proj = nn.Linear(embd_dims, embd_dims)
        self.out_proj = nn.Linear(embd_dims, embd_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B = x.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls_token, x], dim=1)          # [B, N+1, D]

        Q = self.q_proj(x).view(B, -1, self.num_heads, self.dk).transpose(1, 2)
        K = self.k_proj(x).view(B, -1, self.num_heads, self.dk).transpose(1, 2)
        V = self.v_proj(x).view(B, -1, self.num_heads, self.dk).transpose(1, 2)

        attn = torch.softmax((Q @ K.transpose(-2, -1)) / (self.dk ** 0.5), dim=-1)
        attn = self.dropout(attn)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, -1, self.embd_dims)
        out = self.out_proj(out)
        out.unsqueeze(1)
        # return only CLS token’s output (fused representation)
        return out[:, 0, :]  # [B, 768]



def test_clip():
    # Create dummy input [B, N, D]
    inputs = torch.randn(2, 6, 768)   # batch=2, 6 tokens, embedding dim=768

    # Create model instance
    model = SAttention(num_heads=8)  # use same args as in your class definition

    # Forward pass
    output = model(inputs)

    print(f"Input shape:  {inputs.shape}")
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    test_clip()
