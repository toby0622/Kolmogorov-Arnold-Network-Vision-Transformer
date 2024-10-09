import torch
import torch.nn as nn


class GaussianKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_gaussians):
        super(GaussianKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.num_gaussians = num_gaussians

        # 初始化高斯函數的中心 (mu) 和標準差 (sigma)
        self.mu = nn.Parameter(torch.empty(input_dim, num_gaussians))
        self.log_sigma = nn.Parameter(
            torch.empty(input_dim, num_gaussians)
        )  # 使用 log_sigma 以保證 sigma 為正

        # 初始化參數
        nn.init.uniform_(self.mu, -1.0, 1.0)  # 根據需要調整範圍
        nn.init.constant_(self.log_sigma, 0.0)  # 初始 sigma 為 1

        # 初始化線性組合的係數
        self.coeffs = nn.Parameter(torch.empty(input_dim, output_dim, num_gaussians))
        nn.init.normal_(self.coeffs, mean=0.0, std=1 / (input_dim * num_gaussians))

    def forward(self, x):
        """
        前向傳播：
        1. 將輸入 x 正規化到 [-1, 1]（根據需要，可以調整或移除此步驟）。
        2. 計算高斯基底函數。
        3. 將基底函數與係數相乘並求和，生成輸出。
        """
        # 將輸入正規化到 [-1, 1]，根據需要可以調整
        x = torch.tanh(x)

        # 將 x 重塑為 (batch_size, inputdim, 1) 以便與 mu 和 sigma 進行廣播運算
        x = x.view(-1, self.inputdim, 1)  # shape: (batch_size, inputdim, 1)

        # 計算 sigma，確保其為正數
        sigma = torch.exp(self.log_sigma) + 1e-8  # 防止除以零

        # 計算高斯基底函數：exp(-((x - mu)/sigma)^2)
        gaussians = torch.exp(
            -(((x - self.mu) / sigma) ** 2)
        )  # shape: (batch_size, inputdim, num_gaussians)

        # 計算線性組合：對每個輸出維度進行加權求和
        # chebyshev_coeffs 的形狀為 (inputdim, output_dim, num_gaussians)
        # 使用愛因斯坦求和約定進行批量矩陣乘法
        y = torch.einsum(
            "bid,iod->bo", gaussians, self.coeffs
        )  # shape: (batch_size, outdim)

        return y
