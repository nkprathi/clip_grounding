import torch

def similarity(img_emb, txt_emb, logit_scale=None):
    img = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txt = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    cos = img @ txt.T                       # (B, B), range [-1, 1]
    logits = cos * logit_scale.exp() if logit_scale is not None else None
    return cos, logits
