# Created by Lang Huang (laynehuang@outlook.com)
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from timm.models.layers import trunc_normal_


def get_coordinates3d(d, h, w, device='cpu'):
    coords_d = torch.arange(d, device=device)
    coords_h = torch.arange(h, device=device)
    coords_w = torch.arange(w, device=device)
    coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w]))  # 3, D, H, W
    return coords


class WindowAttention3D(nn.Module):
    r"""
    Window based multi-head self attention (W-MSA) module with relative position bias for 3D data.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The depth, height, and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # D, H, W
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        )  # 2*D-1 * 2*H-1 * 2*W-1, nH

        # get pair-wise relative position index for each token inside the window
        coords = get_coordinates3d(*window_size)  # 3, D, H, W
        coords_flatten = torch.flatten(coords, 1)  # 3, D*H*W
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, D*H*W, D*H*W
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # D*H*W, D*H*W, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  # D*H*W, D*H*W
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, pos_idx=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, D*H*W, D*H*W) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B_, nH, N, N

        # relative position bias
        assert pos_idx.dim() == 3, f"Expect the pos_idx/mask to be a 3-d tensor, but got {pos_idx.dim()}"
        rel_pos_mask = torch.masked_fill(torch.ones_like(mask), mask=mask.bool(), value=0.0)
        pos_idx_m = torch.masked_fill(pos_idx, mask.bool(), value=0).view(-1)
        # relative_position_bias : nW, D*H*W, D*H*W, nH
        relative_position_bias = self.relative_position_bias_table[pos_idx_m].view(-1, N, N, self.num_heads)
        relative_position_bias = relative_position_bias * rel_pos_mask.view(-1, N, N, 1)

        nW = relative_position_bias.shape[0]
        relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous()  # nW, nH, D*H*W, D*H*W
        attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + relative_position_bias.unsqueeze(0)

        # attention mask
        attn = attn + mask.view(1, nW, 1, N, N)
        attn = attn.view(B_, self.num_heads, N, N)

        # normalization
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # aggregation
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


def knapsack(W, wt):
    '''Args:
        W (int): capacity
        wt (tuple[int]): the numbers of elements within each window
    '''
    val = wt
    n = len(val)
    K = [[0 for w in range(W + 1)]
            for i in range(n + 1)]
            
    # Build table K[][] in bottom up manner
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1]
                + K[i - 1][w - wt[i - 1]],
                            K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

    # stores the result of Knapsack
    res = res_ret = K[n][W]

    # stores the selected indices
    w = W
    idx = []
    for i in range(n, 0, -1):
        if res <= 0:
            break
        # Either the result comes from the top (K[i-1][w]) 
        # or from (val[i-1] + K[i-1] [w-wt[i-1]]) as in Knapsack table.
        # If it comes from the latter one, it means the item is included.
        if res == K[i - 1][w]:
            continue
        else:
            # This item is included.
            idx.append(i - 1)
            # Since this weight is included, its value is deducted
            res = res - val[i - 1]
            w = w - wt[i - 1]
    
    return res_ret, idx[::-1]   # make the idx in an increasing order


def group_windows(group_size, num_ele_win):
    '''Greedily apply the DP algorithm to group the elements.
    Args:
        group_size (int): maximal size of the group
        num_ele_win (list[int]): number of visible elements of each window
    Outputs:
        num_ele_group (list[int]): number of elements of each group
        grouped_idx (list[list[int]]): the seleted indeices of each group
    '''
    wt = num_ele_win.copy()
    ori_idx = list(range(len(wt)))
    grouped_idx = []
    num_ele_group = []

    while len(wt) > 0:
        res, idx = knapsack(group_size, wt)
        num_ele_group.append(res)

        # append the selected idx
        selected_ori_idx = [ori_idx[i] for i in idx]
        grouped_idx.append(selected_ori_idx)

        # remaining idx
        wt = [wt[i] for i in range(len(ori_idx)) if i not in idx]
        ori_idx = [ori_idx[i] for i in range(len(ori_idx)) if i not in idx]

    return num_ele_group, grouped_idx


class GroupingModule3D:
    def __init__(self, window_size, shift_size, group_size=None):
        # window_size: (D, H, W)
        self.window_size = window_size
        self.shift_size = shift_size
        # TODO change this code
        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size."
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size."
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size."

        self.group_size = group_size or (self.window_size[0] * self.window_size[1] * self.window_size[2])
        self.attn_mask = None
        self.rel_pos_idx = None

    def _get_group_id(self, coords):
        group_id = coords.clone()  # (B, N_vis, 3), N_vis = D*H*W
        # self.window_size: (4,4,4), self.shift_size: (2,2,2)
        # TODO change this code if window_size and shift_size are different at each dimension
        #    for example, window_size[0] != window_size[1] != window_size[2]
        #    assume window_size[0] = window_size[1] = window_size[2], shift_size[0] = shift_size[1] = shift_size[2]
        # Adjust group_id by the shift size
        group_id[:, :, 0] += (self.window_size[0] - self.shift_size[0]) % self.window_size[0]
        group_id[:, :, 1] += (self.window_size[1] - self.shift_size[1]) % self.window_size[1]
        group_id[:, :, 2] += (self.window_size[2] - self.shift_size[2]) % self.window_size[2]

        # Calculate the group id by window size
        # group_id[:, :, 0] = group_id[:, :, 0] // self.window_size[0]
        # group_id[:, :, 1] = group_id[:, :, 1] // self.window_size[1]
        # group_id[:, :, 2] = group_id[:, :, 2] // self.window_size[2]
        group_id[:, :, 0] = torch.div(group_id[:, :, 0], self.window_size[0], rounding_mode='trunc')
        group_id[:, :, 1] = torch.div(group_id[:, :, 1], self.window_size[1], rounding_mode='trunc')
        group_id[:, :, 2] = torch.div(group_id[:, :, 2], self.window_size[2], rounding_mode='trunc')

        # Combine the group ids from each dimension into a single group id
        group_id = (group_id[0, :, 0] * (group_id[0, :, 1].max() + 1) * (group_id[0, :, 2].max() + 1) +
                    group_id[0, :, 1] * (group_id[0, :, 2].max() + 1) +
                    group_id[0, :, 2])  # (N_vis,)

        return group_id

    def _get_attn_mask(self, group_id):
        pos_mask = (group_id == -1)
        pos_mask = torch.logical_and(pos_mask[:, :, None], pos_mask[:, None, :])
        gid = group_id.float()
        attn_mask_float = gid.unsqueeze(2) - gid.unsqueeze(1)
        attn_mask = torch.logical_or(attn_mask_float != 0, pos_mask)
        attn_mask_float.masked_fill_(attn_mask, -100.)
        return attn_mask_float

    def _get_rel_pos_idx(self, coords):
        # num_groups, group_size, group_size, 3
        rel_pos_idx = coords[:, :, None, :] - coords[:, None, :, :]
        rel_pos_idx[:, :, :, 0] += self.window_size[0] - 1
        rel_pos_idx[:, :, :, 1] += self.window_size[1] - 1
        rel_pos_idx[:, :, :, 2] += self.window_size[2] - 1
        rel_pos_idx[..., 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        rel_pos_idx[..., 1] *= 2 * self.window_size[2] - 1
        rel_pos_idx = rel_pos_idx.sum(dim=-1)
        return rel_pos_idx

    def _prepare_masking(self, coords):
        # coords: (B, N_vis, 3)
        group_id = self._get_group_id(coords)   # (N_vis, )
        attn_mask = self._get_attn_mask(group_id.unsqueeze(0))
        rel_pos_idx = self._get_rel_pos_idx(coords[:1])

        # do not shuffle
        self.idx_shuffle = None
        self.idx_unshuffle = None

        return attn_mask, rel_pos_idx

    def _prepare_grouping(self, coords):
        # find out the elements within each local window
        # coords: (B, N_vis, 3)
        group_id = self._get_group_id(coords)   # (N_vis, )
        idx_merge = torch.argsort(group_id)
        group_id = group_id[idx_merge].contiguous()
        exact_win_sz = torch.unique_consecutive(group_id, return_counts=True)[1].tolist()

        # group the windows by DP algorithm  # FIXME
        self.group_size = min(self.window_size[0] * self.window_size[1] * self.window_size[1], max(exact_win_sz))
        num_ele_group, grouped_idx = group_windows(self.group_size, exact_win_sz)

        # pad the splits
        idx_merge_spl = idx_merge.split(exact_win_sz)
        group_id_spl = group_id.split(exact_win_sz)
        shuffled_idx, attn_mask = [], []
        for num_ele, gidx in zip(num_ele_group, grouped_idx):
            pad_r = self.group_size - num_ele
            # shuffle indices: (group_size)
            sidx = torch.cat([idx_merge_spl[i] for i in gidx], dim=0)
            shuffled_idx.append(F.pad(sidx, (0, pad_r), value=-1))
            # attention mask: (group_size)
            amask = torch.cat([group_id_spl[i] for i in gidx], dim=0)
            attn_mask.append(F.pad(amask, (0, pad_r), value=-1))

        # shuffle indices: (num_groups * group_size,)
        self.idx_shuffle = torch.cat(shuffled_idx, dim=0)
        # unshuffle indices that exclude the padded indices: (N_vis, )
        self.idx_unshuffle = torch.argsort(self.idx_shuffle)[-sum(num_ele_group):]
        self.idx_shuffle[self.idx_shuffle==-1] = 0  # index_select does not permit negative index

        # attention mask: (num_groups, group_size, group_size)
        attn_mask = torch.stack(attn_mask, dim=0)
        attn_mask = self._get_attn_mask(attn_mask)

        # relative position indices: (num_groups, group_size, group_size)
        coords_shuffled = coords[0][self.idx_shuffle].reshape(-1, self.group_size, 3)
        rel_pos_idx = self._get_rel_pos_idx(coords_shuffled) # num_groups, group_size, group_size
        rel_pos_mask = torch.ones_like(rel_pos_idx).masked_fill_(attn_mask.bool(), 0)
        rel_pos_idx = rel_pos_idx * rel_pos_mask

        return attn_mask, rel_pos_idx

    def prepare(self, coords, num_tokens):
        if num_tokens <= 2 * self.window_size[0] * self.window_size[1] * self.window_size[2]:
            self._mode = 'masking'
            return self._prepare_masking(coords)
        else:
            self._mode = 'grouping'
            return self._prepare_grouping(coords)

    def group(self, x):
        if self._mode == 'grouping':
            self.ori_shape = x.shape
            x = torch.index_select(x, 1, self.idx_shuffle)   # (B, nG*GS, C)
            x = x.reshape(-1, self.group_size, x.shape[-1]) # (B*nG, GS, C)
        return x

    def merge(self, x):
        if self._mode == 'grouping':
            B, N, C = self.ori_shape
            x = x.reshape(B, -1, C) # (B, nG*GS, C)
            x = torch.index_select(x, 1, self.idx_unshuffle)    # (B, N, C)
        return x


if __name__ == '__main__':
    # check the correctness of the grouping module
    window_size = (4, 7, 7)
    shift_size = (0, 0, 0)
    D, H, W = 8, 14, 14
    coords = get_coordinates3d(D, H, W, device='cpu').reshape(3, -1).permute(1, 0).unsqueeze(0)  # (N_vis, 3)
    gm = GroupingModule3D(window_size, shift_size)
    group_ids = gm._get_group_id(coords)
    group_ids2 = group_ids.reshape(D, H, W)
    print(group_ids)