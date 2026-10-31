import numpy as np
import pandas as pd
import os
from progress.bar import Bar
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from scipy.ndimage import label
import time
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import multiprocessing as mtp
import random
import math
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 设计规则与合法性求解器 (保持不变)
# ==========================================

class DesignRules:
    CANVAS_SIZE = 2048.0
    MIN_SPACE = 45.0
    MIN_WIDTH = 45.0
    MIN_AREA = 2025.0
    MAX_AREA = 3000000.0


def solve_paper_legality(topology_matrix, rules=DesignRules):
    rows, cols = topology_matrix.shape
    x0 = np.concatenate([np.full(cols, rules.CANVAS_SIZE / cols), np.full(rows, rules.CANVAS_SIZE / rows)])
    bounds = Bounds(1.0, rules.CANVAS_SIZE)
    A_eq_x, A_eq_y = np.zeros((1, cols + rows)), np.zeros((1, cols + rows))
    A_eq_x[0, :cols], A_eq_y[0, cols:] = 1, 1
    linear_sum_constraint = LinearConstraint(np.vstack([A_eq_x, A_eq_y]), [rules.CANVAS_SIZE, rules.CANVAS_SIZE],
                                             [rules.CANVAS_SIZE, rules.CANVAS_SIZE])
    A_ineq, lb_ineq, ub_ineq = [], [], []

    def extract_1d_constraints(line, offset_idx):
        if len(line) == 0: return
        prev_idx = 0
        changes = np.where(line[:-1] != line[1:])[0] + 1
        boundaries = np.append(changes, len(line))
        for b_idx in boundaries:
            segment_len = b_idx - prev_idx
            val = line[prev_idx]
            row = np.zeros(cols + rows)
            row[offset_idx + prev_idx: offset_idx + b_idx] = 1
            limit = rules.MIN_WIDTH if val == 1 else rules.MIN_SPACE
            A_ineq.append(row)
            lb_ineq.append(limit)
            ub_ineq.append(np.inf)
            prev_idx = b_idx

    for r in range(rows): extract_1d_constraints(topology_matrix[r, :], 0)
    for c in range(cols): extract_1d_constraints(topology_matrix[:, c], cols)

    constraints = [linear_sum_constraint]
    if A_ineq: constraints.append(LinearConstraint(np.array(A_ineq), lb_ineq, ub_ineq))

    labeled_array, num_features = label(topology_matrix)
    if num_features > 0:
        feature_indices = [(np.where(labeled_array == i)[0], np.where(labeled_array == i)[1]) for i in
                           range(1, num_features + 1)]

        def area_func(v):
            dx, dy = v[:cols], v[cols:]
            return np.array([np.sum(dx[xs] * dy[ys]) for ys, xs in feature_indices])

        constraints.append(NonlinearConstraint(area_func, np.full(num_features, rules.MIN_AREA),
                                               np.full(num_features, rules.MAX_AREA)))

    def objective(v):
        return np.var(v[:cols]) + np.var(v[cols:])

    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 50, 'ftol': 1e-3, 'disp': False})

        # ========【修改点1：不仅返回是否成功，还要返回 dx 和 dy】========
        if res.success:
            return True, res.x[:cols], res.x[cols:]
        else:
            return False, None, None
    except:
        return False, None, None


def generate_msgdata(p):
    p = (p > 0.5).astype(int)
    # ========【修改点2：空数组时返回 6 个元素】========
    if p.size == 0: return ['', 0, 0, 0, "", ""]

    if p.shape[0] > 0:
        idx = [0]
        for i in range(1, len(p)):
            if not np.array_equal(p[i], p[i - 1]): idx.append(i)
        p = p[idx]
    if p.shape[1] > 0:
        idx = [0]
        for i in range(1, len(p[0])):
            if not np.array_equal(p[:, i], p[:, i - 1]): idx.append(i)
        p = p[:, idx]

    if p.size == 0: return ['', 0, 0, 0, "", ""]
    if p.shape[0] > 0 and np.all(p[0] == 0): p = p[1:]
    if p.shape[0] > 0 and np.all(p[-1] == 0) and random.random() > 0.5: p = p[:-1]
    if p.shape[1] > 0 and np.all(p[:, 0] == 0): p = p[:, 1:]
    if p.shape[1] > 0 and np.all(p[:, -1] == 0): p = p[:, :-1]

    if p.size == 0: return ['', 0, 0, 0, "", ""]

    # ========【修改点3：接收并解析 dx 和 dy】========
    dx_str, dy_str = "", ""
    valid = 0

    if p.shape[0] > 1 and p.shape[1] > 1:
        # 接收修改后的求解器返回的 3 个变量
        is_legal, dx, dy = solve_paper_legality(p)
        if is_legal:
            valid = 1
            # 将 numpy 数组转换为保留一位小数的逗号分隔字符串，例如 "45.0,90.0,45.0"
            dx_str = ','.join([f"{v:.1f}" for v in dx])
            dy_str = ','.join([f"{v:.1f}" for v in dy])
    else:
        # 尺寸太小不交给求解器，原代码默认 valid=1
        valid = 1

    topo_str = ''.join(map(str, p.flatten()))
    # 返回列表增加 dx_str 和 dy_str
    return [topo_str, p.shape[1], p.shape[0], valid, dx_str, dy_str]


# ==========================================
# [改动1] 神经网络组件 — DiT (Diffusion Transformer) 替换 UNet
# ==========================================

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning on timestep."""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        # AdaLN: output 6 params (scale, shift, gate) × 2 (for attn + mlp)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

        # Zero-initialize the final linear in each modulation branch
        nn.init.zeros_(self.adaLN[1].weight)
        nn.init.zeros_(self.adaLN[1].bias)

    def forward(self, x, c):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN(c).chunk(6, dim=-1)

        # Self-attention with AdaLN
        xn = self.norm1(x) * (1.0 + scale_attn.unsqueeze(1)) + shift_attn.unsqueeze(1)
        x = x + gate_attn.unsqueeze(1) * self.attn(xn, xn, xn)[0]

        # MLP with AdaLN
        xn = self.norm2(x) * (1.0 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(xn)
        return x


class SparseDiT(nn.Module):
    """[Stage 3] 双分支 DiT — 拓扑头 + 几何头共享 backbone。
    Patchifies (B, C, H, W) folded input into tokens, processes with
    Transformer blocks, and unpatchifies back to pixel logits + geometry map.
    """
    def __init__(self, in_channels=16, hidden_size=384, depth=6, num_heads=6,
                 patch_size=4, num_classes=2):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_patches = (32 // patch_size) ** 2
        self.patch_dim = in_channels * patch_size ** 2

        # Patch embedding
        self.patch_embed = nn.Linear(self.patch_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # DiT blocks (shared backbone)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        # ── 拓扑头 ──
        self.topo_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.adaLN_topo = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        nn.init.zeros_(self.adaLN_topo[1].weight)
        nn.init.zeros_(self.adaLN_topo[1].bias)
        self.topo_proj = nn.Linear(hidden_size, self.patch_dim * num_classes)

        # ── 几何头：P(G|T) 串联，输入 = 共享特征 + 拓扑信息 ──
        self.geo_norm = nn.LayerNorm(hidden_size * 2, elementwise_affine=False)
        self.adaLN_geo = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size * 2))
        nn.init.zeros_(self.adaLN_geo[1].weight)
        nn.init.zeros_(self.adaLN_geo[1].bias)
        # 拓扑嵌入：将拓扑信息映射为 token 维度
        self.topo_embed = nn.Linear(self.patch_dim * num_classes, hidden_size)
        # 几何投影：输入 = shared_features + topology_embedding (2 * hidden_size)
        self.geo_proj = nn.Linear(hidden_size * 2, self.patch_dim * 1)

    def _unpatchify(self, tokens, out_ch, B, C, H, W):
        p = self.patch_size
        np_h, np_w = H // p, W // p
        x = tokens.view(B, np_h, np_w, out_ch, C, p, p)
        x = x.permute(0, 3, 4, 1, 5, 2, 6).contiguous()
        return x.reshape(B, out_ch, C, H, W)

    def forward(self, x, t):
        B, C, H, W = x.shape
        p = self.patch_size

        # Patchify
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, self.num_patches, self.patch_dim)

        # Embed + pos
        x = self.patch_embed(x) + self.pos_embed

        # Timestep
        c = self.time_embed(t)

        # Shared backbone
        for blk in self.blocks:
            x = blk(x, c)

        # ── 拓扑头：先预测拓扑 P(T) ──
        s_t, g_t = self.adaLN_topo(c).chunk(2, dim=-1)
        x_t = self.topo_norm(x) * (1.0 + s_t.unsqueeze(1)) + g_t.unsqueeze(1)
        topo_tokens = self.topo_proj(x_t)               # (B, 64, patch_dim*2) = (B, 64, 512)
        logits = self._unpatchify(topo_tokens, self.num_classes, B, C, H, W)

        # ── 几何头：P(G|T) 串联，输入 = 共享特征 + 拓扑嵌入 ──
        # .detach() 截断 L_geo 回流到拓扑头，保证拓扑独立训练
        topo_info = self.topo_embed(topo_tokens.detach())  # (B, 64, hidden_size)
        x_geo_input = torch.cat([x, topo_info], dim=-1)  # (B, 64, 2 * hidden_size)

        s_g, g_g = self.adaLN_geo(c).chunk(2, dim=-1)
        x_g = self.geo_norm(x_geo_input) * (1.0 + s_g.unsqueeze(1)) + g_g.unsqueeze(1)
        geo_map = self._unpatchify(self.geo_proj(x_g), 1, B, C, H, W)

        return logits, geo_map


# ============================================================
# [改动2] 离散扩散模型 — Sparse + 空间稳态先验 D3PM + DiT + 无约束
#
# D3PM（状态 {0=空白, 1=金属}），前向转移矩阵:
#   Q_t = α_t·I + (1-α_t)·1·π(i,j)^T
#   其中 π(i,j) 是「空间稳态先验」：每个 folded 位置的金属先验概率，
#   从训练数据统计得到（居中填充后中心金属密度高、四周低）。
#   —— 相比标准 D3PM 的全局常数 π=[p0,p1]，这里把稳态分布推广为
#      数据驱动的空间密度场，前向收敛到版图的空间先验而非均匀噪声。
#   前向: q(x_t|x_0) = x_0·(ᾱ_t + (1-ᾱ_t)·π) + (1-x_0)·((1-ᾱ_t)·π)
#   后验: q(x_{t-1}|x_t, x_0) ∝ x_t^T·Q_t ⊙ (x_0^T·Q̅_{t-1})
#   采样: 从 π(i,j) 场采样起点，DiT 预测 x_0 logits，温度 τ 调节多样性
# ============================================================

class DiffusionModel:
    def __init__(self, init_lr=2e-4, max_iter=500000, batch_size=128, img_size=32, img_channel=16, sv_itr=5000,
                 show_itr=100, gpu_id='0', model_path='D:\\wyz\\deepattern-master\\models\\iccad_kongjianxianyan',
                 timesteps=1000, fast_sample_m=5, sample_temp=8.5,
                 prior_strength=1.0):
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
            print(f"[GPU] Using {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("[WARN] CUDA not available, falling back to CPU")
        self.img_size, self.model_path = img_size, model_path
        self.max_iter, self.batch_size, self.sv_itr, self.show_itr = max_iter, batch_size, sv_itr, show_itr

        self.squish_block_size = 4
        self.C = 16
        self.folded_size = 16

        self.timesteps, self.num_classes, self.fast_m = timesteps, 2, fast_sample_m
        self.lambda_loss = 0.001
        self.sample_temp = sample_temp       # 采样温度：>1 增加多样性
        if not 0.0 <= prior_strength <= 1.0:
            raise ValueError("prior_strength must be in [0, 1]")
        self.prior_strength = prior_strength  # 0=标准全局先验，1=完全数据驱动空间先验

        # D3PM: α_t = 1 - β_t，α̅_t = Π α_t（累积信噪比）
        # 更宽的 beta 范围 → 更强的正向扰动 → 模型学到更丰富的分布
        self.betas = np.linspace(1e-4, 0.03, self.timesteps, dtype=np.float32)
        self.alphas = 1.0 - self.betas          # 单步保留概率
        self.alphas_bar = np.cumprod(self.alphas)  # 累积保留概率 α̅_t
        self.alphas_bar = np.insert(self.alphas_bar, 0, 1.0)  # α̅_0 = 1

        # D3PM 稀疏稳态分布: π = [p0, p1]
        self.p1 = 0.0013
        self.p0 = 1.0 - self.p1

        # 空间稳态先验 π(i,j)：从训练数据统计的每个 folded 位置的金属频率。
        # 相比标准 D3PM 的全局常数 p1，这里用数据驱动的空间密度场作为稳态分布。
        self.pi = None                       # (16, 32, 32) 空间先验，run() 统计 / test() 加载

        # [改动2] DiT 替换 UNet（加大容量捕捉更多模式）
        self.model = SparseDiT(in_channels=self.C, hidden_size=512, depth=8, num_heads=8,
                               patch_size=4, num_classes=self.num_classes).to(self.device)
        print(f"[DiT] Params: {sum(p.numel() for p in self.model.parameters()):,}")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=init_lr)

    def unfold_np(self, x):
        B, H, W, C_fold = x.shape
        b = self.squish_block_size
        return x.reshape(B, H, W, b, b, 1).transpose(0, 1, 3, 2, 4, 5).reshape(B, H * b, W * b, 1)

    def _compute_stationary(self, data, smooth=1.0):
        """统计并收缩空间先验，兼顾位置结构与跨数据集稳定性。"""
        cache = data.data_cache                       # (N, 16, 32, 32)
        n = cache.shape[0]
        metal_count = cache.sum(axis=0)

        # 二状态 Laplace 平滑：金属和空白类别各加 smooth。
        empirical_pi = (metal_count + smooth) / (n + 2.0 * smooth)

        # 将经验空间场向标准 D3PM 全局稀疏先验收缩，避免数据集特有的
        # 绝对位置分布完全支配前向扩散和反向采样。
        pi = self.p1 + self.prior_strength * (empirical_pi - self.p1)
        return np.clip(pi, 1e-8, 1.0 - 1e-8).astype(np.float32)

    def run(self, data, resume=True):
        self.model.train()
        os.makedirs(self.model_path, exist_ok=True)
        sample_dir = os.path.join(self.model_path, 'train_samples')
        os.makedirs(sample_dir, exist_ok=True)

        # 统计并持久化空间稳态先验 π(i,j)（每个 folded 位置的金属频率 + 拉普拉斯平滑）
        if self.pi is None:
            self.pi = self._compute_stationary(data)
        np.save(os.path.join(self.model_path, 'stationary_prior.npy'), self.pi)
        print(f"[Prior] shrinkage spatial pi strength={self.prior_strength:.2f}, "
              f"shape={self.pi.shape}, min={self.pi.min():.5f}, "
              f"max={self.pi.max():.5f}, mean={self.pi.mean():.5f}, "
              f"data_mean={data.data_cache.mean():.5f}")

        start_step = 0
        latest_ckpt_path = os.path.join(self.model_path, 'ddpm_model_latest.pth')

        if resume and os.path.exists(latest_ckpt_path):
            print(f"Loading checkpoint from {latest_ckpt_path}...")
            ckpt = torch.load(latest_ckpt_path, map_location=self.device)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                self.model.load_state_dict(ckpt['model'])
                self.optimizer.load_state_dict(ckpt['optimizer'])
                start_step = ckpt['step'] + 1
                print(f"Resuming training from step {start_step}...")
            else:
                self.model.load_state_dict(ckpt)
                print("Loaded legacy model weights.")

        for step in range(start_step, self.max_iter):
            x0_batch = data.getTrainBatchDP(self.batch_size)
            x0_tensor = torch.tensor((x0_batch > 0.5).astype(np.int64), device=self.device)
            t = torch.randint(1, self.timesteps + 1, (self.batch_size,), device=self.device)

            alpha_bar_t = torch.tensor(self.alphas_bar, device=self.device)[t].view(-1, 1, 1, 1)
            pi = torch.tensor(self.pi, device=self.device).unsqueeze(0)  # (1,16,32,32) → 广播

            # 空间稳态先验前向：q(x_t|x_0) = x_0·(ᾱ_t + (1-ᾱ_t)·π) + (1-x_0)·((1-ᾱ_t)·π)
            prob_1 = x0_tensor.float() * (alpha_bar_t + (1.0 - alpha_bar_t) * pi) + \
                     (1.0 - x0_tensor.float()) * ((1.0 - alpha_bar_t) * pi)

            x_t = (torch.rand_like(x0_tensor.float()) < prob_1).float()

            logits, geo_map = self.model(x_t, t)

            ce_loss = F.cross_entropy(logits, x0_tensor)

            # [Stage 3] 几何头自监督损失：预测局部图案密度
            geo_target = x0_tensor.float().mean(dim=1, keepdim=True)  # (B, 1, 32, 32)
            geo_loss = F.mse_loss(geo_map, geo_target)

            loss = ce_loss * self.lambda_loss + 0.0008 * geo_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            if step % self.show_itr == 0:
                print(f"[{datetime.now()}] Step {step}, CE: {ce_loss.item():.4f}, Geo: {geo_loss.item():.4f}, Total: {loss.item():.6f}")

            if step > 0 and step % self.sv_itr == 0:
                self.model.eval()
                print(f"\n>>> Generating intermediate sample at step {step}...")

                with torch.no_grad():
                    # 从空间稳态先验 π(i,j) 采样起点（结构化先验，而非均匀随机噪声）
                    sample_img = (np.random.rand(4, 16, 32, 32) < self.pi).astype(np.float32)
                    pi = self.pi                      # (16,32,32) → 广播到 (4,16,32,32)
                    fast_m = 10
                    for t_s in range(self.timesteps, 0, -fast_m):
                        t_prev = max(0, t_s - fast_m)
                        t_tensor = torch.full((4,), t_s, device=self.device, dtype=torch.long)

                        s_logits, _ = self.model(torch.tensor(sample_img, device=self.device), t_tensor)
                        p1_pred = F.softmax(s_logits, dim=1).cpu().numpy()[:, 1]
                        p0_pred = 1.0 - p1_pred

                        alpha_T = np.prod(self.alphas[t_prev:t_s]) if t_prev > 0 else self.alphas_bar[t_s]
                        alpha_bar_s = self.alphas_bar[t_prev]

                        # 空间稳态先验下的区间转移矩阵（逐位置，pi 为空间场）
                        Q_00 = alpha_T + (1.0 - alpha_T) * (1.0 - pi)
                        Q_01 = (1.0 - alpha_T) * pi
                        Q_10 = (1.0 - alpha_T) * (1.0 - pi)
                        Q_11 = alpha_T + (1.0 - alpha_T) * pi

                        Qb_00 = alpha_bar_s + (1.0 - alpha_bar_s) * (1.0 - pi)
                        Qb_01 = (1.0 - alpha_bar_s) * pi
                        Qb_10 = (1.0 - alpha_bar_s) * (1.0 - pi)
                        Qb_11 = alpha_bar_s + (1.0 - alpha_bar_s) * pi

                        Q_to_xt_from_1 = np.where(sample_img == 1, Q_11, Q_10)
                        Q_to_xt_from_0 = np.where(sample_img == 1, Q_01, Q_00)

                        # 边缘化 x_0（张量化，两个分支展开）
                        num_1_1 = Q_to_xt_from_1 * Qb_11
                        num_0_1 = Q_to_xt_from_0 * Qb_10
                        post_1_1 = num_1_1 / (num_1_1 + num_0_1 + 1e-8)

                        num_1_0 = Q_to_xt_from_1 * Qb_01
                        num_0_0 = Q_to_xt_from_0 * Qb_00
                        post_1_0 = num_1_0 / (num_1_0 + num_0_0 + 1e-8)

                        prob_tm_1 = p1_pred * post_1_1 + p0_pred * post_1_0
                        sample_img = (np.random.rand(*sample_img.shape) < prob_tm_1).astype(np.float32)

                    sample_img_bhwc = np.transpose(sample_img, (0, 2, 3, 1))
                    gen_64 = self.unfold_np(sample_img_bhwc)[:, :, :, 0]
                preview_path = os.path.join(sample_dir, f'step_{step}_preview.png')

                fig, axes = plt.subplots(2, 2, figsize=(8, 8))
                for i, ax in enumerate(axes.flat):
                    ax.imshow(gen_64[i], cmap='gray', interpolation='none', vmin=0, vmax=1)
                    ax.axis('off')
                plt.savefig(preview_path)
                plt.close()
                print(f">>> Preview saved to {preview_path}")

                ckpt_dict = {
                    'step': step,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }
                torch.save(ckpt_dict, latest_ckpt_path)

                if step % 10000 == 0:
                    step_ckpt_path = os.path.join(self.model_path, f'ddpm_model_step_{step}.pth')
                    torch.save(ckpt_dict, step_ckpt_path)
                    print(f">>> Checkpoint saved to {step_ckpt_path}")
                print("\n")
                self.model.train()

    def test(self, data_adapter):
        out_dir = os.path.join(self.model_path, 'test/')
        os.makedirs(out_dir, exist_ok=True)
        ckpt_path = os.path.join(self.model_path, 'ddpm_model_latest.pth')

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                self.model.load_state_dict(ckpt['model'])
                print(f"Loaded weights from step {ckpt.get('step', 'unknown')}")
            else:
                self.model.load_state_dict(ckpt)
        else:
            print("WARNING: Checkpoint not found! Generating with untrained weights.")

        self.model.eval()

        # 加载训练时统计的空间稳态先验 π；缺失则退回全局 p1
        prior_path = os.path.join(self.model_path, 'stationary_prior.npy')
        if os.path.exists(prior_path):
            self.pi = np.load(prior_path)
            print(f"Loaded stationary prior pi from {prior_path}, shape={self.pi.shape}")
        else:
            self.pi = np.full((self.C, self.img_size, self.img_size), self.p1, dtype=np.float32)
            print("WARNING: stationary prior not found, falling back to global p1")

        p_pool = mtp.Pool(4)

        pi = self.pi                      # (16,32,32) 空间稳态先验

        for b_idx in range(1000):
            bar = Bar(f'Sampling Batch {b_idx}', max=self.timesteps // self.fast_m)
            # 从空间稳态先验 π(i,j) 采样起点（结构化先验）
            img = (np.random.rand(100, 16, 32, 32) < pi).astype(np.float32)

            for t_s in range(self.timesteps, 0, -self.fast_m):
                t_prev = max(0, t_s - self.fast_m)
                with torch.no_grad():
                    logits, _ = self.model(torch.tensor(img, device=self.device),
                                           torch.full((100,), t_s, device=self.device, dtype=torch.long))

                    # D3PM 后验采样 + 温度调节
                    temperature = self.sample_temp    # τ > 1 → 更均匀分布 → 更多样
                    p1_pred = F.softmax(logits / temperature, dim=1).cpu().numpy()[:, 1]
                    p0_pred = 1.0 - p1_pred

                alpha_T = np.prod(self.alphas[t_prev:t_s]) if t_prev > 0 else self.alphas_bar[t_s]
                alpha_bar_s = self.alphas_bar[t_prev]

                # 空间稳态先验下的区间转移矩阵（逐位置，pi 为空间场）
                Q_00 = alpha_T + (1.0 - alpha_T) * (1.0 - pi)
                Q_01 = (1.0 - alpha_T) * pi
                Q_10 = (1.0 - alpha_T) * (1.0 - pi)
                Q_11 = alpha_T + (1.0 - alpha_T) * pi

                Qb_00 = alpha_bar_s + (1.0 - alpha_bar_s) * (1.0 - pi)
                Qb_01 = (1.0 - alpha_bar_s) * pi
                Qb_10 = (1.0 - alpha_bar_s) * (1.0 - pi)
                Qb_11 = alpha_bar_s + (1.0 - alpha_bar_s) * pi

                Q_to_xt_from_1 = np.where(img == 1, Q_11, Q_10)
                Q_to_xt_from_0 = np.where(img == 1, Q_01, Q_00)

                # 边缘化 x_0（张量化，两个分支展开）
                num_1_1 = Q_to_xt_from_1 * Qb_11
                num_0_1 = Q_to_xt_from_0 * Qb_10
                post_1_1 = num_1_1 / (num_1_1 + num_0_1 + 1e-8)

                num_1_0 = Q_to_xt_from_1 * Qb_01
                num_0_0 = Q_to_xt_from_0 * Qb_00
                post_1_0 = num_1_0 / (num_1_0 + num_0_0 + 1e-8)

                prob_tm_1 = p1_pred * post_1_1 + p0_pred * post_1_0
                img = (np.random.rand(*img.shape) < prob_tm_1).astype(np.float32)

                # [改动4] 采样噪声注入 — 每步随机翻转 0.7% 像素，打破确定性轨迹（按空间先验采样）
                noise_mask = np.random.rand(*img.shape) < 0.007
                img = np.where(noise_mask, np.random.rand(*img.shape) < pi, img).astype(np.float32)

                bar.next()
            bar.finish()

            img_bhwc = np.transpose(img, (0, 2, 3, 1))
            gen_64 = self.unfold_np(img_bhwc)[:, :, :, 0]
            results = p_pool.map(generate_msgdata, gen_64)
            pd.DataFrame(results, columns=['topoSig', 'cX', 'cY', 'valid', 'dx', 'dy']).to_pickle(
                f"{out_dir}noise_data_{b_idx}.pkl")

        p_pool.close()
        p_pool.join()