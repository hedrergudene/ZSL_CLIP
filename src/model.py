# Requirements
import torch
from transformers import AutoConfig, AutoModel


# Utils


## Linear Block & normalisation
class ProjectionHead(torch.nn.Module):
    def __init__(self,
                 input_dim:int,
                 output_dim:int,
                 dropout:float=.25,
                 device:str='cuda:0',
                 ):
        super(ProjectionHead, self).__init__()
        # Parameters
        self.projection = torch.nn.Sequential(torch.nn.Dropout(dropout),
                                              torch.nn.Linear(input_dim, output_dim, device=device),
                                        )

    def forward(self, x):
        x = self.projection(x)
        return x/x.norm(p=2, dim=-1, keepdim=True)


# Model
class CLIPModel(torch.nn.Module):
    def __init__(self,
                 vision_backbone:str,
                 text_backbone:str,
                 freeze_backbone:bool=False,
                 dim:int=256,
                 dropout:float=.25,
                 device:str='cuda:0',
                 ):
        super(CLIPModel, self).__init__()

        # Take pretrained nlp model from custom config
        nlp_config = AutoConfig.from_pretrained(text_backbone)
        nlp_config.update(
            {
                "output_hidden_states": False,
                "add_pooling_layer": True,
            }
        )
        self.nlp_transformer = AutoModel.from_config(nlp_config).to(device)

        # Take pretrained vision model from custom config
        vision_config = AutoConfig.from_pretrained(vision_backbone)
        vision_config.update(
            {
                "output_last_hidden_state": False,
                "add_pooling_layer": True,
            }
        )
        self.vision_transformer = AutoModel.from_config(vision_config).to(device)
        # Backbones freeze (if needed)
        if freeze_backbone:
            for param in self.nlp_transformer.parameters():
                param.requires_grad = False
            for param in self.vision_transformer.parameters():
                param.requires_grad = False
        # Allignment layers
        self.nlp_linear = ProjectionHead(nlp_config.hidden_size, dim, dropout, device)
        self.vision_linear = ProjectionHead(vision_config.hidden_size, dim, dropout, device)
        # Extra parameters
        self.device = device
    
    def forward(self, image, input_ids, attention_mask):
        # Input
        nlp_feats = self._disentangle_nlp_transformer(input_ids, attention_mask)
        vision_feats = self._disentangle_vision_transformer(image)
        # Standardise shape and values
        text_embeddings = self.nlp_linear(nlp_feats)
        vision_embeddings = self.vision_linear(vision_feats)
        # Get logits
        logits_per_text = text_embeddings @ vision_embeddings.T
        # Output (we only return text ones, since vision logits are the transpose
        return logits_per_text
   

   
    def _disentangle_nlp_transformer(self,
                                     input_ids:torch.Tensor,
                                     attention_mask:torch.Tensor,
                                     ):
        # Get HuggingFace output
        output = self.nlp_transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Check that output has the desired attribute
        if not hasattr(output, 'pooler_output'):
            raise AttributeError(f"Transformers output does not have the attribute 'pooler_output'. Please check imported backbone.")
        # Return pooler_output
        return output.pooler_output


    def _disentangle_vision_transformer(self,
                                        img_batch:torch.Tensor,
                                        ):
        # Get HuggingFace output
        output = self.vision_transformer(img_batch)
        # Check that output has the desired attribute
        if not hasattr(output, 'pooler_output'):
            raise AttributeError(f"Transformers output does not have the attribute 'pooler_output'. Please check imported backbone.")
        # Return pooler_output
        return output.pooler_output
