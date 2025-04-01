import sys
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat as loadmat
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = False
from einops import rearrange, repeat
import torch.nn.functional as F
from torch import einsum
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = False
cudnn.enabled = False
import h5py
from sklearn.decomposition import PCA

random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

passed_dataset_name = sys.argv[1]
passed_model_path = sys.argv[2]


def loadData(dataset_name):
    dirpath="/workspaces/hyper-gcn/datasets/HOOO.mat"
    if dataset_name.lower() == "houston2013":
        data_file = f"{dirpath}/Houston13.mat"
        gt_file = f"{dirpath}/Houston13_7gt.mat"
    elif dataset_name.lower() == "houston2018":
        data_file = f"{dirpath}/Houston18.mat"
        gt_file = f"{dirpath}/Houston18_7gt.mat"
        
        # load data
        with h5py.File(data_file, 'r') as data_f:
            X = data_f['ori_data'][()]
        
        # loading ground truth
        with h5py.File(gt_file, 'r') as gt_f:
            y = gt_f['map'][()]
        
    elif dataset_name.lower() == "muufl":
        data_file = r"/workspaces/hyper-gcn/datasets/MUUFL_data.mat"
        
        data = loadmat(data_file)
        X = data['hsi_img'][()]
        y = data['labels'][()]
        y[y == -1] = 0  # convert -1 labels to 0
    else:
        raise ValueError("Invalid dataset_name specified. Use 'Houston2013', 'Houston2018', or 'MUUFL'.")
    
    return X, y

X, y = loadData(passed_dataset_name)

class PatchSet(Dataset):
    """Generate 3D patches from a hyperspectral dataset."""

    def __init__(self, data, gt, patch_size, is_pred=False):
        """
        Initialize the PatchSet dataset.

        Args:
            data (ndarray): 3D hyperspectral image.
            gt (ndarray): 2D array of labels.
            patch_size (int): Size of the 3D patch.
            is_pred (bool): Whether to create data without labels for prediction. Default is False.
        """
        super(PatchSet, self).__init__()
        self.is_pred = is_pred
        self.patch_size = patch_size
        p = self.patch_size // 2

        # Padding the data and ground truth arrays
        self.data = np.pad(data, pad_width=((p, p), (p, p), (0, 0)), mode='constant', constant_values=0)
        """
        pad_width: A tuple specifying the padding width for each dimension. In this case, (p, p) is used for both the first and second
                   dimensions (rows and columns), and (0, 0) is used for the third dimension (channels). This means that p pixels will be added to
                   the beginning and end of each row and column, and no padding will be added to the channels dimension.

        mode='constant': Specifies the padding mode. Here, it's set to 'constant', which means that values will be padded with a constant value.

        constant_values=0: Specifying the constant value that will be used for padding.
        """
        if is_pred:
            gt = np.ones_like(gt)
        self.label = np.pad(gt, pad_width=(p, p), mode='constant', constant_values=0)
        """
        pad_width: A tuple specifying the padding width for each dimension. Here, (p, p) is used, indicating that p pixels will be added to
                   the beginning and end of each row and column.
        """

        # Get indices of non-zero values in ground truth
        x_pos, y_pos = np.nonzero(gt)
        x_pos, y_pos = x_pos + p, y_pos + p  # adjust indices after padding
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])

        # Randomly shuffle indices if not predicting
        if not is_pred:
            np.random.shuffle(self.indices)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.indices)

    def __getitem__(self, i):
        """Get a sample from the dataset."""
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        # Extract the patch and label from the data and ground truth arrays
        data = self.data[x1:x2, y1:y2]
        label = self.label[x, y]

        # Transpose the data to match PyTorch format
        data = np.asarray(data, dtype='float32').transpose((2, 0, 1))
        label = np.asarray(label, dtype='int64')

        # Convert data and label to PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        # Return data and label (or just data if predicting)
        if self.is_pred:
            return data
        else:
            return data, label


PATCH_SIZE = 8
BATCH_SIZE = 64

all_data = PatchSet(X, y, PATCH_SIZE,is_pred = True)
all_loader = DataLoader(all_data,BATCH_SIZE,shuffle= False)

BAND = 64
CLASSES_NUM = 11

# print('-----Importing Setting Parameters-----')
PATCH_LENGTH = 5
lr, num_epochs, batch_size = 0.001, 200, 32

img_rows = 2 * PATCH_LENGTH + 1
# print(img_rows)
img_cols = 2 * PATCH_LENGTH + 1

def name_with_msg(instance: nn.Module, msg: str) -> str:
    return f"[{instance.__class__.__name__}] {msg}"

class CoAtNetRelativeAttention(nn.Module):
    """
    CoAtNetRelativeAttention implements multi-head self-attention with relative positional biases.
    
    Args:
        in_dim (int): Each token is represented as a vector of size `in_dim`.
        pre_height (int): Predefined input height.
        pre_width (int): Predefined input width.
        heads (int, optional): Number of attention heads.
        kv_dim (int, optional): Dimension of key and value projections (default: same as `in_dim`).
        head_dim (int, optional): Dimension of each attention head. 
        proj_dim (int, optional): Projection dimension.
        out_dim (int, optional): Output feature dimension (default: same as `in_dim`).
        attention_dropout (float, optional): Dropout rate for attention weights (default: 0.0).
        ff_dropout (float, optional): Dropout rate for the output projection (default: 0.0).
        use_bias (bool, optional): Whether to use bias in linear projections (default: False).
        dtype (torch.dtype, optional): Data type for numerical precision (default: torch.float32).
    """
    def __init__(
        self, 
        in_dim: int,
        pre_height: int,
        pre_width: int,
        *,
        heads: int = None, 
        kv_dim: int = None, 
        head_dim: int = None, 
        proj_dim: int = None,
        out_dim: int = None,
        attention_dropout: float = 0.0, 
        ff_dropout: float = 0.0, 
        use_bias: bool = False,
        dtype=torch.float32, 
        **rest
    ):
        super().__init__()

        # Determine embedding dimension for projections
        dim = proj_dim if proj_dim is not None else in_dim
        out_dim = out_dim if out_dim is not None else in_dim

        # Ensure either `heads` or `head_dim` is provided
        assert (
            heads is not None or head_dim is not None
        ), f"[{self.__class__.__name__}] Either `heads` or `head_dim` must be specified."

        # Derive the number of heads and head dimension if one is missing
        self.heads = heads if heads is not None else dim // head_dim
        head_dim = head_dim if head_dim is not None else dim // self.heads

        # Ensure that the total embedding size matches the product of `heads` and `head_dim`
        assert (
            head_dim * self.heads == dim
        ), f"[{self.__class__.__name__}] Head dimension times the number of heads must be equal to embedding dimension (`in_dim` or `proj_dim`)"
        
        # Define linear layers for Query (Q), Key (K), and Value (V)
        self.Q = nn.Linear(in_dim, dim, bias=use_bias)
        self.K = nn.Linear(kv_dim if kv_dim is not None else in_dim, dim, bias=use_bias)
        self.V = nn.Linear(kv_dim if kv_dim is not None else in_dim, dim, bias=use_bias)
        
        # Output projection layer
        self.out_linear = nn.Linear(dim, out_dim)

        # Dropout layers for attention weights and final output
        self.attention_dropout = nn.Dropout2d(attention_dropout)
        self.out_dropout = nn.Dropout(ff_dropout)

        # Scaling factor for dot-product attention
        self.scale = head_dim ** (-0.5)

        # Mask value for masked attention (very large negative number to nullify masked entries)
        self.mask_value = -torch.finfo(dtype).max  # Pytorch default float type

        # Relative positional encoding parameters
        self.pre_height = pre_height
        self.pre_width = pre_width

        # Define learnable relative bias parameter
        self.relative_bias = nn.Parameter(
            torch.randn(self.heads, int((2*pre_height - 1)*(2*pre_width - 1))),
            requires_grad=True
        )
        
        # Compute relative indices for position encoding
        self.register_buffer("relative_indices", self._get_relative_indices(pre_height, pre_width))

    def forward(self, x, attention_mask=None, verbose=False):
        """
        Forward pass for multi-head attention with relative positional encoding.

        Args:
            x (torch.Tensor or tuple of torch.Tensor): 
                - Shape: (batch_size, channels, height, width) for spatial inputs
                - or (batch_size, seq_len, in_dim) for sequence inputs
            attention_mask (torch.Tensor, optional): Attention mask for sequence inputs
            verbose (bool, optional): If True, print intermediate tensor shapes

        Returns:
            torch.Tensor: Output of the multi-head attention module
        """
        h = self.heads  # Number of attention heads
        
        # Check if input is spatial (4D) or sequential (3D)
        is_spatial = len(x.shape) == 4

        if is_spatial:
            b, c, H, W = x.shape
            # Reshape input for attention mechanism
            x = rearrange(x, "b c h w -> b (h w) c")
        else:
            b, seq_len, c = x.shape
            H, W = seq_len, 1

        if verbose:
            print(f"Input shape after initial processing: {x.shape}")

        # Compute Query, Key, and Value projections
        q, k, v = map(lambda proj: proj(x), (self.Q, self.K, self.V))

        if verbose:
            print(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")

        # Reshape projections to (batch_size, heads, seq_len, head_dim)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        if verbose:
            print(f"Q reshaped: {q.shape}, K reshaped: {k.shape}, V reshaped: {v.shape}")

        # Scale the key vectors
        k = k * self.scale

        # Determine relative biases based on input size
        if H == self.pre_height and W == self.pre_width:
            relative_indices = self.relative_indices
            relative_bias = self.relative_bias
        else:
            relative_indices = self._get_relative_indices(H, W)
            relative_bias = self._interpolate_relative_bias(H, W)

        # Expand indices and biases to match batch and head dimensions
        relative_indices = repeat(relative_indices, "n m -> b h n m", b=b, h=h)
        relative_bias = repeat(relative_bias, "h r -> b h n r", b=b, n=H*W)

        # Gather and apply relative biases
        relative_biases = relative_bias.gather(dim=-1, index=relative_indices)

        # Compute attention scores
        attention = einsum("b h n d, b h m d -> b h n m", q, k) + relative_biases

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = repeat(attention_mask, "b 1 n m -> b h n m", h=h)
            attention.masked_fill_(attention_mask, self.mask_value)

        # Apply softmax to get attention weights
        attention = attention.softmax(dim=-1)
        
        # Apply dropout to attention weights
        attention = self.attention_dropout(attention)
        
        # Compute weighted sum of value vectors
        out = einsum("b h n m, b h m d -> b h n d", attention, v)
        
        # Reshape output to (batch_size, seq_len, heads * head_dim)
        out = rearrange(out, "b h n d -> b n (h d)")
        
        # Apply final linear projection
        out = self.out_linear(out)

        # Apply output dropout
        out = self.out_dropout(out)

        # Reshape back to spatial if input was spatial
        if is_spatial:
            out = rearrange(out, "b (h w) c -> b c h w", h=H, w=W)

        return out

    def _get_relative_indices(self, height: int, width: int) -> torch.Tensor:
        """
        Compute relative position indices for attention biases.
        
        Args:
            height (int): Height of the input feature map.
            width (int): Width of the input feature map.
        
        Returns:
            torch.Tensor: Tensor of relative indices.
        """
        height, width = int(height), int(width)
        ticks_y, ticks_x = torch.arange(height), torch.arange(width)
        grid_y, grid_x = torch.meshgrid(ticks_y, ticks_x)
        out = torch.empty(height*width, height*width).fill_(float("nan"))

        for idx_y in range(height):
            for idx_x in range(width):
                rel_indices_y = grid_y - idx_y + height
                rel_indices_x = grid_x - idx_x + width
                flatten_indices = (rel_indices_y*width + rel_indices_x).flatten()
                out[idx_y*width + idx_x] = flatten_indices

        # Ensure there are no NaN or negative values in indices
        assert not out.isnan().any(), "`relative_indices` contain NaN values"
        assert (out >= 0).all(), "`relative_indices` contain negative indices"
        
        return out.to(torch.long)

    def _interpolate_relative_bias(self, height: int, width: int) -> torch.Tensor:
        """
        Interpolate the relative bias for varying input sizes.
        
        Args:
            height (int): Target height.
            width (int): Target width.
        
        Returns:
            torch.Tensor: Interpolated relative bias.
        """
        out = rearrange(self.relative_bias, "h (n m) -> 1 h n m", n=(2*self.pre_height - 1))
        out = nn.functional.interpolate(out, size=(2*height - 1, 2*width - 1), mode="bilinear", align_corners=True)
        return rearrange(out, "1 h n m -> h (n m)")

class AttentionGCN(nn.Module):
    def __init__(self, pre_height, pre_width, in_dim, proj_dim, head_dim, n_classes, attention_dropout, ff_dropout, verbose=False):
        super(AttentionGCN, self).__init__()
        
        # Input: (batch_size, in_channels=17, 11, 11)
        self.conv1 = nn.Conv2d(17, pre_height, kernel_size=1)  # Output: (batch_size, 8, 11, 11)
        self.bn1 = nn.BatchNorm2d(pre_height, eps=1e-3)
        
        # CoAtNetRelativeAttention layer replacing the placeholder
        self.attention = CoAtNetRelativeAttention(
            pre_height=pre_height,
            pre_width=pre_width,
            in_dim=in_dim,
            proj_dim=proj_dim,
            head_dim=head_dim,
            attention_dropout=attention_dropout,
            ff_dropout=ff_dropout,
            
        )  # Expected output: (batch_size, 11, 11, 11)
        
        
        # Graph reasoning module 1
        self.squeezer = nn.Conv1d(pre_height**2, 16, kernel_size=1)  # Output: (batch_size, 16, 11)
        self.gconv = nn.Conv1d(16, pre_height**2, kernel_size=1)  # Output: (batch_size, 121, 11)
        self.unsqueezer = nn.Conv1d(pre_height**2, pre_height**2, kernel_size=1)  # Output: (batch_size, 121, 11)
        
        self.conv2 = nn.Conv2d(4*pre_height, 4*pre_height, kernel_size=1)  # Output: (batch_size, 32, 11, 11)
        self.bn2 = nn.BatchNorm2d(4*pre_height, eps=1e-3)
        
        # Graph reasoning module 2
        self.squeezer2 = nn.Conv1d(pre_height**2, pre_height**2, kernel_size=1)  # Output: (batch_size, 64, 121)
        self.gconv2 = nn.Conv1d(4*pre_height, 4*pre_height, kernel_size=1)  # Output: (batch_size, 44, 121)
        self.unsqueezer2 = nn.Conv1d(4*pre_height, pre_height**2, kernel_size=1)  # Output: (batch_size, 121, 121)
        
        # Additional convolutions
        self.conv3 = nn.Conv2d(136, 32, kernel_size=1)  # Output: (batch_size, 32, 11, 11)
        self.bn3 = nn.BatchNorm2d(32, eps=1e-3)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: (batch_size, 64, 11, 11)
        self.bn4 = nn.BatchNorm2d(64, eps=1e-3)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=1)  # Output: (batch_size, 128, 11, 11)
        self.bn5 = nn.BatchNorm2d(128, eps=1e-3)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * pre_height * pre_height, 100)
        self.fc2 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, n_classes)

    def forward(self, x, verbose = False):
        
        batch_size = x.shape[0]
        if verbose:
            print("Input shape:", x.shape)  # (batch_size, 17, 11, 11)
        
        x = F.relu(self.conv1(x))
        if verbose:
            print("After conv1:", x.shape)  # (batch_size, 11, 11, 11)
            # plot_batch_hist(x, "Before BN1")

        x = self.bn1(x)
        if verbose:
            print("After bn1:", x.shape)  # (batch_size, 8, 11, 11)
            # plot_batch_hist(x, "After BN1")
        
        attention_out = self.attention(x)
        if verbose:
            print("After attention:", attention_out.shape)  # (batch_size, 11, 11, 11)
        
        # Graph reasoning module 1
        graph_t = x.view(batch_size,8, 8*8).permute(0,2,1)
        if verbose:
            print("Graph input shape:", graph_t.shape)  # (batch_size, 121, 11)
        
        squeezed_graph_t = F.relu(self.squeezer(graph_t))
        if verbose:
            print("After squeezer:", squeezed_graph_t.shape)  # (batch_size, 16, 11)
        
        gconv = F.relu(self.gconv(squeezed_graph_t))
        if verbose:
            print("After gconv:", gconv.shape)  # (batch_size, 121, 11)
        
        unsqueezed_graph = F.relu(self.unsqueezer(gconv))
        if verbose:
            print("After unsqueezer:", unsqueezed_graph.shape)  # (batch_size, 121, 11)
        
        glore = unsqueezed_graph.view(batch_size, 8, 8, 8)
        if verbose:
            print("After glore reshape:", glore.shape)  # (batch_size, 11, 11, 11)
        
        block1 = torch.cat([x, glore], dim=1)
        if verbose:
            print("After block1 concat:", block1.shape)  # (batch_size, 22, 11, 11)

        # Graph module 1 ends
        
        block12 = torch.cat([x, block1, attention_out], dim=1)
        if verbose:
            print("After block12 concat:", block12.shape)  # (batch_size, 44, 11, 11)
        
        x = F.relu(self.conv2(block12))
        if verbose:
            print("After conv2:", x.shape)  # (batch_size, 44, 11, 11)
            # plot_batch_hist(x,"Before BN2")

        x = self.bn2(x)
        if verbose:
            print("After bn2:", x.shape)
            # plot_batch_hist(x, "After BN2")

        # Graph Module 2 Start
        graph_t2 = x.view(batch_size,32,8*8).permute(0,2,1) #(b,121,44)
        squeezed_graph_t2 = F.relu(self.squeezer2(graph_t2)) #(b,121, 44)
        squeezed_graph2 = squeezed_graph_t2.permute(0,2,1) #(b,44,121)
        gconv2 = F.relu(self.gconv2(squeezed_graph2)) #(b,44,121)
        unsqueezed_graph_t2 = F.relu(self.unsqueezer2(gconv2)) #(b,121,121)
        glore2 = unsqueezed_graph_t2.view(batch_size,64,8,8)

        if verbose:
            print("After glore2:", glore2.shape)
        
        # Graph Module 2 end

        block2 = torch.cat([x, glore2], dim=1) ## b,165,11,11

        block21 = torch.cat([x, block2, attention_out], dim=1) ## b, 220, 11,11

        if verbose:
            print("After block21 concat:", block21.shape)

        x = F.relu(self.conv3(block21))
        if verbose:
            print("After conv3:", x.shape)
            # plot_batch_hist(x, "Before BN3")

        x = self.bn3(x)
        if verbose:
            print("After bn3:", x.shape)
            plot_batch_hist(x, "After BN3")

        x = F.relu(self.conv4(x))
        if verbose:
            print("After conv4:", x.shape)  # (batch_size, 64, 11, 11)
            plot_batch_hist(x, "Before BN4")

        x = self.bn4(x)
        if verbose:
            print("After bn4:", x.shape)
            plot_batch_hist(x, "After BN4")

        x = F.relu(self.conv5(x))
        if verbose:
            print("After conv5:", x.shape)  # (batch_size, 128, 11, 11)
            plot_batch_hist(x, "Before BN5")

        x = self.bn5(x)
        if verbose:
            print("After bn5:", x.shape)
            plot_batch_hist(x, "After BN5")

        
        x = x.view(batch_size, -1)
        if verbose:
            print("After flattening:", x.shape)  # (batch_size, 128 * 11 * 11)
        
        x = F.relu(self.fc1(x))
        if verbose:
            print("After fc1:", x.shape)  # (batch_size, 100)
        
        x = F.relu(self.fc2(x))
        if verbose:
            print("After fc2:", x.shape)  # (batch_size, 20)
        
        x = F.softmax(self.fc3(x), dim=1)
        if verbose:
            print("After fc3 (softmax):", x.shape)
            # print(x)# (batch_size, 7)
        
        return x

pre_height = 8
pre_width = 8
in_dim = 8
proj_dim = 8
head_dim = 4
n_classes = 11
attention_dropout = 0.1
ff_dropout = 0.1


model = AttentionGCN(pre_height, pre_width, in_dim, proj_dim, head_dim, n_classes, attention_dropout, ff_dropout)

def apply_pca(hsi, out_components=17):
    b, c, h, w = hsi.shape  # Batch, Channels, Height, Width

    # Reshape to (b, h*w, c) for PCA
    reshaped_hsi = hsi.reshape(b, c, h * w).permute(0, 2, 1).cpu().numpy()  # Convert to NumPy (b, h*w, c)

    # Fit PCA on the first sample, transform all samples
    pca = PCA(n_components=out_components)
    pca.fit(reshaped_hsi[0])  # Fit only on the first batch element
    transformed_hsi = np.array([pca.transform(sample) for sample in reshaped_hsi])  # Transform all

    # Convert to PyTorch tensor and reshape back
    output_hsi = torch.from_numpy(transformed_hsi).float().permute(0, 2, 1).reshape(b, out_components, h, w)

    return output_hsi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = model.to(device)

model = torch.load(passed_model_path, map_location = device)


def list_to_colormap(x_list):
    """Convert a label array to an RGB colormap."""
    colormap = np.zeros((x_list.shape[0], x_list.shape[1], 3))  # Create RGB array
    
    # Define class colors (example: 10 classes, change as needed)
    colors = {
        0: [255, 0, 0],   # Red
        1: [0, 255, 0],   # Green
        2: [0, 0, 255],   # Blue
        3: [255, 255, 0], # Yellow
        4: [255, 165, 0], # Orange
        5: [128, 0, 128], # Purple
        6: [0, 255, 255], # Cyan
        7: [255, 192, 203], # Pink
        8: [128, 128, 0], # Olive
        9: [0, 128, 128], # Teal
    }

    # Convert labels to colors
    for (i, j), label in np.ndenumerate(x_list):  
        colormap[i, j] = np.array(colors.get(label, [0, 0, 0])) / 255.  # Default black for unknown labels

    return colormap

def predict_and_save_grid(dataset, model, save_path):
    """
    Process each patch sequentially, predict its label, and map it back to the (325, 220) grid.

    Args:
        dataset (PatchSet): The dataset containing hyperspectral patches.
        model (torch.nn.Module): The trained model.
        save_path (str): Path to save the prediction map.
    """
    model.eval()
    device = next(model.parameters()).device  # Ensure model and data are on the same device

    preds = np.zeros(len(dataset), dtype=np.uint8)  # Store predictions

    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]  # Unpack (data, label)
            data = data.unsqueeze(0)
            data = apply_pca(data).to(device)
            output = model(data)
            pred = output.argmax(dim=1).cpu().item()  # Get class prediction
            preds[i] = pred  # Store prediction

    # Reshape predictions into a 2D grid (325, 220)
    pred_grid = preds.reshape((325, 220))

    # Convert to colormap and save
    colormap = list_to_colormap(pred_grid)
    plt.imsave(save_path, colormap)

output_path = os.path.join("live_results", f"{passed_dataset_name}_live_results.png")
predict_and_save_grid(all_data, model, "prediction_map.png")