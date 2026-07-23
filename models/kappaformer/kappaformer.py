import logging
import time
import math
import numpy as np
import torch
import torch.nn as nn
from pyexpat.model import XML_CQUANT_OPT
from torch_geometric.nn import GlobalAttention

try:
    from e3nn import o3
except ImportError:
    pass

from .gaussian_rbf import GaussianRadialBasisLayer
from torch.nn import Linear
from .edge_rot_mat import init_edge_rot_mat
from .so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2
)
from .module_list import ModuleListInfo
from .so2_ops import SO2_Convolution
from .radial_function import RadialFunction
from .layer_norm import (
    EquivariantLayerNormArray, 
    EquivariantLayerNormArraySphericalHarmonics, 
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer
)

from .transformer_block import (
    SO2EquivariantGraphAttention,
    FeedForwardNetwork,
    TransBlockV2, 
)

from .input_block import EdgeDegreeEmbedding
from ..base_model import ModelOutput
from .ocpmodels import BaseModel

from e3nn.io import CartesianTensor
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super(GaussianSmearing, self).__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = (
            -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        )
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
    

# Statistics of IS2RE 100K 
_AVG_NUM_NODES  = 77.81317
_AVG_DEGREE     = 23.395238876342773    # IS2RE: 100k, max_radius = 5, max_neighbors = 100


###################### Inv attention ###############################
from torch_scatter import scatter_add,scatter_sum,scatter_softmax
import math


###################### GRU-based fusion ###############################
class Fusion(nn.Module):
    def __init__(self, in_channel, h_channel, out_channel):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_channel)
        self.norm2 = nn.LayerNorm(in_channel)
        self.cell = nn.GRUCell(in_channel, h_channel)
        self.fc = nn.Linear(h_channel, out_channel)

    def forward(self, x, h):
        x = self.norm1(x)
        h = self.norm2(h)
        h = self.cell(x, h)
        out = self.fc(h)
        return h, out
    

########################### MoE block #######################################
class Gate(nn.Module):
    def __init__(self, in_dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_experts, bias=False)
    def forward(self, x):
        logits = self.fc(x)              # [batch, num_experts]
        weights = F.softmax(logits, dim=-1)
        return weights
    
class Expert(nn.Module):
    def __init__(self, in_channel, h_channel, drop_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channel, h_channel * 4),
            nn.SiLU(),
            nn.Dropout(drop_rate),
            nn.Linear(h_channel * 4, h_channel * 4),
            nn.SiLU(),
            nn.Dropout(drop_rate),
            nn.Linear(h_channel * 4, in_channel)
        )
    def forward(self, x):
        return self.net(x)  

class MoEBlock(nn.Module):
    def __init__(self, in_channel, h_channel, num_experts=4, drop_rate=0.2):
        super().__init__()
        self.num_experts = num_experts
        self.gate_harm = Gate(in_channel, num_experts)
        self.gate_anharm = Gate(in_channel, num_experts)
        self.experts = nn.ModuleList([Expert(in_channel, h_channel, drop_rate) for _ in range(num_experts)])
        self.norm_harm = nn.LayerNorm(in_channel)
        self.norm_anharm = nn.LayerNorm(in_channel)
        # Add
        # self.proj = nn.Linear(in_channel*2, h_channel)

    def forward(self, x1=None, x2=None):
        if x2 is None:
            h1 = self.norm_harm(x1)
            gates_harm = self.gate_harm(h1)
            out_harm = 0
            for i in range(self.num_experts):
                expert_out = self.experts[i](h1)
                out_harm += expert_out * gates_harm[:, i].unsqueeze(1)
            outadd_harm = x1 + out_harm
            return outadd_harm
        else:
            # For anharm branch
            # Add harm and anharm features
            # x_all = self.proj(torch.cat([x1, x2], dim=-1))
            # x_all = x1 + x2
            h2 = self.norm_anharm(x2)
            gates_anharm = self.gate_anharm(h2)
            out_anharm = 0
            for i in range(self.num_experts):
                expert_out = self.experts[i](h2)
                out_anharm += expert_out * gates_anharm[:, i].unsqueeze(1)
            outadd_anharm = x2 + out_anharm
            return outadd_anharm   
        
class FFN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_channel, out_channel * 4),
            nn.SiLU(),
            nn.Linear(out_channel * 4, out_channel)
        )
        self.norm = nn.LayerNorm(in_channel)
    def forward(self, x):
        out = x + self.ffn(self.norm(x))
        return out

# harm || anharm fusion
# class PhyFusion(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super().__init__()
#         self.W1 = nn.Linear(in_channel, out_channel, bias=False)
#         self.W2 = nn.Linear(in_channel, out_channel, bias=True)


#     def forward(self, x1, x2):
#         z = torch.sigmoid(self.W1(x1) + self.W2(x2))
#         out = (1.0 - z) * x1 + z * x2
#         return out
# class PhyFusion(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super().__init__()
#         self.proj_w = nn.Linear(in_channel, out_channel)
#         self.proj_b = nn.Linear(in_channel, out_channel)
#         self.act = nn.GLU()
#         nn.init.constant_(self.proj_w.weight, 1)
#         nn.init.constant_(self.proj_b.weight, 0)

#     def forward(self, x1, x2):
#         w = self.proj_w(x1)
#         b = self.proj_b(x1)
#         out = self.act(w * x2 + b)
#         return out
        

############################ harm anharm constant ############################
def cal_harm(B, G, rho):
    GPa2SI = 1.0e9
    B = B*GPa2SI
    G = G*GPa2SI
    # Calculate velocity
    v_l = torch.sqrt((B+4.0*G/3.0)/rho)
    v_t = torch.sqrt(G/rho)
    v_a = ((1.0/3.0)*(v_l**(-3.0)+2.0*v_t**(-3.0)))**(-1.0/3.0)
    harm = v_a**3.0
    return torch.log10(harm)

def cal_gamma(B1, G1, B2, G2, V1, V2):
    GPa2SI = 1.0e9
    B1 = B1*GPa2SI
    B2 = B2*GPa2SI
    G1 = G1*GPa2SI
    G2 = G2*GPa2SI
    volume = V1
    V_2 = V2
    # volume in unit of angstrom^{-3}
    Gru_l = (-1) * 0.5 * (volume*1.0e-30 / ((B1 + 4 / 3 * G1)* 10**(9))) * (((B1 + 4 / 3 * G1) - (B2 + 4 / 3 * G2))* 10**(9) / (volume*1.0e-30 - V_2*1.0e-30)) - 1 / 6
    Gru_t = (-1) * 0.5 * volume*1.0e-30 / (G1* 10**(9)) * (G1 - G2)* 10**(9) / (volume*1.0e-30 - V_2*1.0e-30) - 1 / 6
    gamma = torch.sqrt((Gru_l ** 2 + 2 * Gru_t ** 2) / 3)
    A = 2.43e-6/(1.0-0.514/gamma + 0.228/(gamma**2))
    anharm = A / gamma**2
    return anharm

def cal_anharm(gamma):
    A = 2.43e-6/(1.0-0.514/gamma + 0.228/(gamma**2))
    anharm = A / gamma**2
    return torch.log10(anharm)

def cal_constant(V, natoms, Ma):
    # Planck constant: (J.s)
    h = 6.62607004e-34
    # h = 6.6260695729 * 10 ** (-34) / (2 * np.pi)
    # Boltzmann constant (J/K = W*s/K)
    kb = 1.38064852e-23
    t=300
    volume = V
    # delta in angstrom
    delta = (volume/natoms)**(1.0/3.0)
    constant = Ma*delta*natoms**(-2.0/3.0)*(h/kb*(3.0*natoms/(4.0*torch.pi*volume*1.0e-30))**(1.0/3.0))**3.0/(t)
    return torch.log10(constant)


################### Kappaformer ##############################
class Kappaformer(BaseModel):
    def __init__(
        self,
        enable_kappa = False,
        use_pbc=True,
        otf_graph=True,
        max_neighbors=20,
        max_radius=10,  
        max_num_elements=95,

        # num_layers=1,
        sphere_channels=256,    #384
        attn_hidden_channels=64,
        num_heads=32,   # 32
        attn_alpha_channels=64,
        attn_value_channels=16,
        ffn_hidden_channels=128, # without FFN in Eqv
        
        norm_type='layer_norm_sh',
        
        lmax_list=[6],  
        mmax_list=[6],  # 6
        grid_resolution=18, # 18 8 24

        # num_sphere_samples=128, #128
        edge_channels=256,
        use_atom_edge_embedding=True, 
        share_atom_edge_embedding=False,
        use_m_share_rad=False,
        distance_function="gaussian",
        num_distance_basis=64,  #128

        attn_activation='silu',
        use_s2_act_attn=False, 
        use_attn_renorm=True,
        ffn_activation='silu',
        use_gate_act=False,
        use_grid_mlp=True, 
        use_sep_s2_act=True,

        alpha_drop=0.2,  # 0.2
        drop_path_rate=0.2,  # 0.2
        proj_drop=0,  # 0

        weight_init='normal',
        label_keys=['B_VRH','G_VRH'],
        return_embedding = False
    ):
        super().__init__()
        self.enable_kappa = enable_kappa
        self.label_keys = label_keys
        self.use_pbc = use_pbc
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.cutoff = max_radius
        self.max_num_elements = max_num_elements

        # self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type
        
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        # self.num_sphere_samples = num_sphere_samples

        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding 
        self.share_atom_edge_embedding = share_atom_edge_embedding
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act
        
        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop

        self.weight_init = weight_init
        assert self.weight_init in ['normal', 'uniform']

        self.device = 'cpu' #torch.cuda.current_device()

        self.grad_forces = False
        self.num_resolutions = len(self.lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        
        # Weights for message initialization
        self.sphere_embedding = nn.Embedding(self.max_num_elements, self.sphere_channels_all)
        
        self.return_embedding = return_embedding
        
        # Initialize the function used to measure the distances between atoms
        assert self.distance_function in [
            'gaussian',
        ]
        if self.distance_function == 'gaussian':
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                self.num_distance_basis,
                0.5
            )
            #self.distance_expansion = GaussianRadialBasisLayer(num_basis=self.num_distance_basis, cutoff=self.max_radius)
        else:
            raise ValueError
        
        # Initialize the sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [self.edge_channels] * 2
        # [num_guassian, channel, channel]

        # Initialize atom edge embedding. # share_atom_edge_embedding=False, Skip
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None
        
        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo('({}, {})'.format(max(self.lmax_list), max(self.lmax_list)))
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l, 
                        m, 
                        resolution=self.grid_resolution, 
                        normalization='component'
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        # Edge-degree embedding
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            rescale_factor=_AVG_DEGREE
        )

        # For loss weights
        # self.log_vars = nn.Parameter(torch.zeros(2))
        # Initialize the blocks for each layer of EquiformerV2
        self.block = TransBlockV2(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels, 
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.num_distance_basis, # Add
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop, 
                self.drop_path_rate,
                self.proj_drop
            )
        
        #in_channel, h_channel, n_heads, max_atoms, num_basis
        # Feature fusion
        self.fusion_harm = Fusion(self.sphere_channels * 2,self.sphere_channels * 2,self.sphere_channels)
        self.fusion_anharm = Fusion(self.sphere_channels * 2,self.sphere_channels * 2,self.sphere_channels)

        # MMoE block
        self.moe_block = MoEBlock(self.sphere_channels,self.sphere_channels)
        # self.ffn_block = FFN(self.sphere_channels,self.sphere_channels)
        # Global attention pool
        # Harm
        self.pool_harm = GlobalAttention(torch.nn.Sequential(nn.Linear(self.sphere_channels, 1, bias=False)))
        # self.head_B = nn.Sequential(
        #     nn.Linear(self.sphere_channels, self.sphere_channels),
        #     nn.SiLU(),
        #     nn.Linear(self.sphere_channels, self.sphere_channels),
        #     nn.SiLU(),
        #     nn.Linear(self.sphere_channels, 1), 
        #     nn.Softplus()
        # )
        self.head_B = nn.Sequential(
            nn.Linear(self.sphere_channels, 1),
            nn.Softplus()
        )

        # self.head_G = nn.Sequential(
        #     nn.Linear(self.sphere_channels, self.sphere_channels),
        #     nn.SiLU(),
        #     nn.Linear(self.sphere_channels, self.sphere_channels),
        #     nn.SiLU(),
        #     nn.Linear(self.sphere_channels, 1), 
        #     nn.Softplus()
        # )
        self.head_G = nn.Sequential(
            nn.Linear(self.sphere_channels, 1),
            nn.Softplus()
        )

        # Anharm
        self.pool_anharm = GlobalAttention(torch.nn.Sequential(nn.Linear(self.sphere_channels, 1, bias=False)))
        # self.phy_fusion = PhyFusion(self.sphere_channels,self.sphere_channels)
        # self.head_anharm = nn.Sequential(
        #     nn.Linear(self.sphere_channels, self.sphere_channels),
        #     nn.SiLU(),
        #     nn.Linear(self.sphere_channels, self.sphere_channels),
        #     nn.SiLU(),
        #     nn.Linear(self.sphere_channels, 1)
        # )
        # self.harm_anharm = nn.Sequential(nn.Linear(self.sphere_channels*2, self.sphere_channels),
        #                                  nn.SiLU())
        self.head_anharm = nn.Sequential(
            nn.Linear(self.sphere_channels, 1)
        )

        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)


    def forward(self, data):
        data.to(self.sphere_embedding.weight.device)
        data.cell = data.cell.view(-1,3,3)
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype
        self.device = data.pos.device

        atomic_numbers = data.atomic_numbers.long()
        # print("atomic_numbers", atomic_numbers)
        num_atoms = len(atomic_numbers)
        pos = data.pos

        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, edge_index, edge_distance_vec
        )
        # print(edge_distance)
        # print(edge_index)
        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        offset = 0
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(
                    atomic_numbers
                    )[:, offset : offset + self.sphere_channels]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Edge encoding (distance and atom edge)
        edge_distance = self.distance_expansion(edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_distance = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            atomic_numbers,
            edge_distance,
            edge_index)
        x.embedding = x.embedding + edge_degree.embedding

        l0_feat_init = x.embedding.narrow(1, 0, 1).squeeze(1) 
        mag_feat_init = x.embedding.norm(dim=1, p=2)
        feat_int = torch.cat([l0_feat_init, mag_feat_init], dim=-1)
        # l0_mag_int = self.fusion_init(feat_int) 
        ###############################################################
        # Update spherical node embeddings
        ###############################################################
        x = self.block(
            x,                  # SO3_Embedding
            atomic_numbers,
            edge_distance,
            edge_index,
            batch=data.batch    # for GraphDropPath
        )
        ## x.embedding.shape [nodes, sph_basis, sph_channel]
        l0_feat = x.embedding.narrow(1, 0, 1).squeeze(1)
        mag_feat = x.embedding.norm(dim=1, p=2)
        feat_eqv = torch.cat([l0_feat, mag_feat], dim=-1)
        l0_mag_h, l0_mag_harm = self.fusion_harm(feat_eqv,feat_int)        

        # x_ffn_harm = self.ffn_block(x=l0_mag_harm)
        x_moe_harm = self.moe_block(x1=l0_mag_harm, x2=None)
        x_pool_harm =self.pool_harm(x_moe_harm, data.batch)
        b = self.head_B(x_pool_harm).squeeze(-1)
        g = self.head_G(x_pool_harm).squeeze(-1)
        
        if self.enable_kappa:
            # breakpoint()
            # print(f'harm:{harm.shape}')
            # TODO: abl sum or last feature
            # anharm = self.attn_pool_anharm(torch.stack(abharm_feats).mean(dim=0), data.batch).squeeze(-1)
            # x_anharm = self.head_x_anharm(x_pool_harm)
            # log_kappa = self.head_anharm(x_anharm).squeeze(-1)
            # l0_feat_harm = l0_mag_harm.narrow(1, 0, 1).squeeze(1)
            # mag_feat_harm = l0_mag_harm.norm(dim=1, p=2)
            # feat_eqv_harm = torch.cat([l0_feat_harm, mag_feat_harm], dim=-1)
            # print("x", x.embedding.shape)
            # print("l0_mag_harm", l0_mag_harm.shape)
            # print("l0_feat_harm", l0_feat_harm.shape)
            # print("mag_feat_harm", mag_feat_harm.shape)
            # print("feat_eqv_harm", feat_eqv_harm.shape)
            # print("feat_int", feat_int.shape)
            l0_mag_h_anharm, l0_mag_anharm =self.fusion_anharm(l0_mag_h,feat_int)
            # h_anharm = self.inv_block_anharm(x=l0_mag_anharm, edge_index=edge_index, edge_rbf=edge_distance, atomic_numbers=atomic_numbers)
            x_moe_anhamrm = self.moe_block(x1=None, x2=l0_mag_anharm)
            x_pool_anharm = self.pool_anharm(x_moe_anhamrm, data.batch)
            # x_all = self.phy_fusion(x_pool_harm, x_pool_anharm)
            # x_all = self.harm_anharm(torch.cat([x_pool_harm, x_pool_anharm], dim=-1))
            
            # x_anharm = self.head_x_anharm(x_pool_anharm)
            gamma = self.head_anharm(x_pool_anharm).squeeze(-1)
            # Physics-informed slack anharm
            Gam = gamma.to(torch.float64)
            y_anharm = cal_anharm(Gam)
            y_anharm = y_anharm.float()
            # print(f'anharm:{anharm.shape}')
            # Physics-informed slack harm
            B = b.to(torch.float64)
            G = g.to(torch.float64)
            rho = data["density"].to(torch.float64)
            y_harm = cal_harm(B, G, rho)
            y_harm = y_harm.float()
            # Constant
            V = data["V1"].to(torch.float64)
            natoms = data["n_atoms"].to(torch.float64)
            Ma = data["Ma"].to(torch.float64)
            constant = cal_constant(V, natoms, Ma)
            constant = constant.float()
            # log_kappa = y_anharm
            log_kappa = y_harm + y_anharm + constant
            if self.return_embedding:
                return b,g,log_kappa,x_pool_harm,x_pool_anharm
            else:
                return b,g,log_kappa
        else:
            if self.return_embedding:
                return b,g,x_pool_harm
            else:
                return b,g
     
    
    def compute_loss(self, pred, batch_data):
        if self.enable_kappa:
            B_pred, G_pred, log_kappa_pred = pred
            B_GT = batch_data['B_VRH']
            G_GT = batch_data['G_VRH']
            # gamma_GT = batch_data['gamma']
            Kappa_log_GT = batch_data['kappa_log']

            B_loss = torch.nn.functional.l1_loss(B_pred,B_GT)
            G_loss = torch.nn.functional.l1_loss(G_pred,G_GT)
            # gamma_loss = torch.nn.functional.l1_loss(gamma_pred,gamma_GT)
            kappa_log_loss = torch.nn.functional.l1_loss(log_kappa_pred,Kappa_log_GT)
            # kappa_log_loss_hb = torch.nn.functional.huber_loss(log_kappa_pred,Kappa_log_GT)
            # loss = B_loss + G_loss + 20 * kappa_log_loss
            loss = B_loss + G_loss + 40 * kappa_log_loss
            # huber loss
            B_loss_huber = torch.nn.functional.huber_loss(B_pred,B_GT)
            G_loss_huber = torch.nn.functional.huber_loss(G_pred,G_GT)
            # gamma_loss_huber = torch.nn.functional.huber_loss(gamma_pred,gamma_GT)
            kappa_log_loss_huber = torch.nn.functional.huber_loss(log_kappa_pred,Kappa_log_GT)
            # loss_huber = B_loss_huber + G_loss_huber + 20 * kappa_log_loss_huber
            loss_huber = B_loss_huber + G_loss_huber + 40 * kappa_log_loss
            bsz = len(batch_data['B_VRH'])
            return ModelOutput(
                loss=loss_huber,
                total_loss = bsz * loss_huber,
                log_output={
                    'B_loss':B_loss,
                    'G_loss':G_loss,
                    'kappa_log_loss':kappa_log_loss,
                    'loss':loss,
                    'B_loss_huber':B_loss_huber,
                    'G_loss_huber':G_loss_huber,
                    'kappa_log_loss_huber':kappa_log_loss_huber,
                    'loss_huber':loss_huber
                },
                num_samples = bsz,
            )
        # if self.enable_kappa:
        #     B_pred, G_pred, log_kappa_pred = pred
        #     B_GT = batch_data['B_VRH']
        #     G_GT = batch_data['G_VRH']
        #     # gamma_GT = batch_data['gamma']
        #     Kappa_log_GT = batch_data['kappa_log']
        #     B_loss = torch.nn.functional.l1_loss(B_pred,B_GT)
        #     G_loss = torch.nn.functional.l1_loss(G_pred,G_GT)
        #     # gamma_loss = torch.nn.functional.l1_loss(gamma_pred,gamma_GT)
        #     kappa_log_loss = torch.nn.functional.l1_loss(log_kappa_pred,Kappa_log_GT)
        #     bsz = len(batch_data['B_VRH'])
        #     return ModelOutput(
        #         loss=kappa_log_loss,
        #         total_loss = bsz * kappa_log_loss,
        #         log_output={
        #             'kappa_log_loss':kappa_log_loss
        #         },
        #         num_samples = bsz
        #     )

        else:
            B_pred, G_pred = pred
            B_GT = batch_data['B_VRH']
            G_GT = batch_data['G_VRH']
            
            B_loss = torch.nn.functional.l1_loss(B_pred,B_GT)
            G_loss = torch.nn.functional.l1_loss(G_pred,G_GT)
            loss = B_loss + G_loss
            # huber loss
            B_loss_huber = torch.nn.functional.huber_loss(B_pred,B_GT)
            G_loss_huber = torch.nn.functional.huber_loss(G_pred,G_GT)
            loss_huber = B_loss_huber + G_loss_huber
            bsz = len(batch_data['B_VRH'])
            return ModelOutput(
                loss=loss_huber,
                total_loss = bsz * loss_huber,
                log_output={
                    'B_loss':B_loss,
                    'G_loss':G_loss,
                    'loss':loss,
                    'B_loss_huber':B_loss_huber,
                    'G_loss_huber':G_loss_huber,
                    'loss_huber':loss_huber
                },
                num_samples = bsz,
            )
        
    def before_batch(self):
        pass
    
    def after_batch(self):
        pass
    
    def ckpt_loaded(self):
        pass
    def reload_checkpoint(self):
        pass


    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)
        

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


    def _init_weights(self, m):
        if (isinstance(m, torch.nn.Linear)
            or isinstance(m, SO3_LinearV2)
        ):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == 'normal':
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    
    def _uniform_init_rad_func_linear_weights(self, m):
        if (isinstance(m, RadialFunction)):
            m.apply(self._uniform_init_linear_weights)


    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear) 
                or isinstance(module, SO3_LinearV2)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormArray)
                or isinstance(module, EquivariantLayerNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonicsV2)
                or isinstance(module, GaussianRadialBasisLayer)):
                for parameter_name, _ in module.named_parameters():
                    if (isinstance(module, torch.nn.Linear)
                        or isinstance(module, SO3_LinearV2)
                    ):
                        if 'weight' in parameter_name:
                            continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)
