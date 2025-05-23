import torch.nn as nn

class CLIP_Adapted_ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        
        self.adapter = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            
            nn.GELU(),
            
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            
            nn.GELU(),
            
            nn.Linear(512, 512),
            nn.LayerNorm(512)
        )
        
    def set_frozen(self):
        for param in self.adapter.parameters():
            param.requires_grad = False
            
    def forward(self, image):
        return self.adapter(self.clip_model.encode_image(image))
    
class CLIP_Adapted_TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.adapter = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            
            nn.GELU(),
            
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            
            nn.GELU(),
            
            nn.Linear(512, 512),
            nn.LayerNorm(512)
        )
    def set_frozen(self):
        for param in self.adapter.parameters():
            param.requires_grad = False
            
    def forward(self, text):
        return self.adapter(self.clip_model.encode_text(text))
    
class DA_adapter(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        for param in clip_model.parameters():
            param.requires_grad = False
        self.ad_imageEncoder = CLIP_Adapted_ImageEncoder(clip_model=clip_model)
        self.ad_textEncoder = CLIP_Adapted_TextEncoder(clip_model=clip_model)
        
    def forward(self, image, text):
        image = self.ad_imageEncoder(image)
        text = self.ad_textEncoder(text)
        return image, text
    
    def get_image_features(self, images):
        return self.ad_imageEncoder(images)
    
    def set_frozen(self):
        self.ad_imageEncoder.set_frozen()
        self.ad_textEncoder.set_frozen()