import torch
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, promptdim):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.qv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        
        self.k = nn.Conv2d(promptdim, dim, kernel_size=1, bias=bias)
        self.k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x, de):
        b,c,h,w = x.shape

        qv = self.qv_dwconv(self.qv(x))
        q,v = qv.chunk(2, dim=1)
        
        k = self.k_dwconv(self.k(de))
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
class GDM(nn.Module):
    def __init__(self, de_dim, dim, stage, degradation_dim=512, activation=nn.PReLU()):
        super(GDM, self).__init__()
        promptdim = int(dim//10*stage)
        self.phi = CrossAttention(dim,2,False,promptdim)
        self.phit = Attention(dim,2,False)
        self.r = nn.Parameter(torch.ones(1))
        self.linear = nn.Linear(degradation_dim, de_dim)
        self.de_dim = de_dim
        
        self.prompt_param = nn.Parameter(torch.rand(1,de_dim,promptdim,int(96//stage),int(96//stage)))
        
    def forward(self, x, img, degradation_vertor):
        b, dim, h, w = x.shape
        if degradation_vertor == "":
            phixsy = self.phi(x) - img
            x = x - self.r*self.phit(phixsy)
        else:
            de = F.softmax(self.linear(degradation_vertor), dim=1).view(b, self.de_dim, 1, 1, 1)
            weights = self.prompt_param.repeat(b,1,1,1,1)
            prompt = de * weights
            prompt = torch.sum(prompt,dim=1)
            prompt = F.interpolate(prompt,(h,w),mode="bilinear")
            phixsy = self.phi(x, prompt) - img
            x = x - self.r*self.phit(phixsy)
        return x
    
class num_Transformer_Block(nn.Module):
    def __init__(self, num_blocks, dim, num_heads, ffn_expansion_factor, bias=False, LayerNorm_type='WithBias'):
        super(num_Transformer_Block, self).__init__()
        self.blocks = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
    
    def forward(self, x):
        return self.blocks(x)
    
class DUN_BaseBlock(nn.Module):
    def __init__(self, de_dim, stage, num_blocks, dim, num_heads, ffn_expansion_factor):
        super(DUN_BaseBlock, self).__init__()
        self.gradient_descent = GDM(de_dim, dim, stage)
        self.denoiser = num_Transformer_Block(num_blocks, dim, num_heads, ffn_expansion_factor)
        
    def forward(self, x, img, degradation_vector):
        x = self.gradient_descent(x, img, degradation_vector)
        x = self.denoiser(x)
        return x, img

##########################################################################
##---------- VLU-Net -----------------------
class VLUNet(nn.Module):
    def __init__(self, 
        de_dim,
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',
        decoder = False,
    ):

        super(VLUNet, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        
        self.patch_embed_i = OverlapPatchEmbed(inp_channels, dim)
        
        self.decoder = decoder
        
        self.encoder_level1 = DUN_BaseBlock(de_dim,stage=1, num_blocks=num_blocks[0], dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor)
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.down1_2_i = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = DUN_BaseBlock(de_dim,stage=2, num_blocks=num_blocks[1], dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor)
                
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.down2_3_i = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = DUN_BaseBlock(de_dim,stage=3, num_blocks=num_blocks[2], dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor)

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.down3_4_i = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = DUN_BaseBlock(de_dim,stage=4, num_blocks=num_blocks[3], dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor)
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.up4_3_i = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.reduce_chan_level3_img = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)

        self.decoder_level3 = DUN_BaseBlock(de_dim,stage=3, num_blocks=num_blocks[2], dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor)

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.up3_2_i = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.reduce_chan_level2_img = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)

        self.decoder_level2 = DUN_BaseBlock(de_dim,stage=2, num_blocks=num_blocks[1], dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor)
                
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.up2_1_i = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = DUN_BaseBlock(de_dim,stage=1, num_blocks=num_blocks[0], dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor)
                
        self.refinement = DUN_BaseBlock(de_dim,stage=1, num_blocks=num_refinement_blocks, dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor)
                    
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, degradation, noise_emb = None):

        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level1_i = self.patch_embed_i(inp_img)
        #
        out_enc_level1, o_img1 = self.encoder_level1(inp_enc_level1, inp_enc_level1_i, degradation)
        
        out_enc_level2, o_img2 = self.encoder_level2(self.down1_2(out_enc_level1), self.down1_2(o_img1), degradation)

        out_enc_level3, o_img3 = self.encoder_level3(self.down2_3(out_enc_level2), self.down2_3_i(o_img2), degradation)
        
        latent, img_l = self.latent(self.down3_4(out_enc_level3), self.down3_4_i(o_img3), degradation)
        
        
        inp_dec_level3 = self.up4_3(latent)
        i_img3 = self.up4_3_i(img_l)
        
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        
        i_img3 = torch.cat([i_img3, o_img3], 1)
        i_img3 = self.reduce_chan_level3_img(i_img3)
        
        inp_dec_level2, i_img2, = self.decoder_level3(inp_dec_level3, i_img3, degradation)


        inp_dec_level2 = self.up3_2(inp_dec_level2)
        i_img2 = self.up3_2_i(i_img2)
        
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        
        i_img2 = torch.cat([i_img2, o_img2], 1)
        i_img2 = self.reduce_chan_level2_img(i_img2)

        inp_dec_level1, i_img1 = self.decoder_level2(inp_dec_level2, i_img2, degradation)
        
        inp_dec_level1 = self.up2_1(inp_dec_level1)
        i_img1 = self.up2_1_i(i_img1)
        
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        i_img1 = torch.cat([i_img1, o_img1], 1)
        
        out, img = self.decoder_level1(inp_dec_level1, i_img1, degradation)

        out, _ = self.refinement(out, img, degradation)

        out = self.output(out) + inp_img

        return out