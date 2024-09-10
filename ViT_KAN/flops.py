import torch
from fvcore.nn import FlopCountAnalysis
from vit_model import Attention


def main():
    # 自注意力機制
    a1 = Attention(dim=512, num_heads=1)
    # 移除 Wo，將注意力機制的輸出直接輸出，轉換為原始維度，簡化計算並更容易分析
    a1.proj = torch.nn.Identity()

    # 多頭注意力機制
    a2 = Attention(dim=512, num_heads=8)

    # [batch_size, num_tokens, total_embed_dim]
    t = (torch.rand(32, 1024, 512),)

    flops1 = FlopCountAnalysis(a1, t)
    print("自注意力機制 FLOPs:", flops1.total())

    flops2 = FlopCountAnalysis(a2, t)
    print("多頭注意力機制 FLOPs:", flops2.total())


if __name__ == "__main__":
    main()
