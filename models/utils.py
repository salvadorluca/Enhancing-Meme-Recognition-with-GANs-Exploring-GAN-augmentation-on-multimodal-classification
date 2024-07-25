import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_gradient_penalty(disc, real_img_embedding, real_text_embedding, fake_img_embedding, fake_text_embedding, device):
    # Compute interpolation
    alpha = torch.rand(real_img_embedding.size(0), 1).to(device)
    img_interpolation = (alpha * real_img_embedding + (1 - alpha) * fake_img_embedding).requires_grad_()
    text_interpolation = (alpha * real_text_embedding + (1 - alpha) * fake_text_embedding).requires_grad_()

    # Compute output
    output = disc(img_interpolation, text_interpolation)

    # Compute gradient
    gradients = autograd.grad(outputs=output,
                              inputs=[img_interpolation, text_interpolation],
                              grad_outputs=torch.ones(output.size()).to(device),
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)

    # Compute gradient penalty
    gradients_img = gradients[0].view(real_img_embedding.size(0), -1)
    gradients_text = gradients[1].view(real_text_embedding.size(0), -1)
    gradient_penalty = ((gradients_img.norm(2, dim=1) - 1) ** 2).mean() + ((gradients_text.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Freeze model
def set_model_require_grad(model, require_grad):
    for param in model.parameters():
        param.requires_grad = require_grad

# Clamp weights
def clamp_weights(model, weight_cliping):
    for p in model.parameters():
        p.data.clamp_(weight_cliping, weight_cliping)

def L1_regularization(params):
    x = torch.cat([p.view(-1) for p in params])
    return x.abs().mean() 

def L2_regularization(params):
    x = torch.cat([p.view(-1) for p in params])
    return x.pow(2).mean()

def relation_consistency_loss(fake_img_embedding, fake_text_embedding, real_img_embedding, real_text_embedding):
    # Normalize features
    fake_cos_sim_img = F.normalize(fake_img_embedding, p=2, dim=-1)
    real_cos_sim_img = F.normalize(real_img_embedding, p=2, dim=-1)
    fake_cos_sim_text = F.normalize(fake_text_embedding, p=2, dim=-1)
    real_cos_sim_text = F.normalize(real_text_embedding, p=2, dim=-1)

    # Compute cosine similarity matrices
    fake_cos_sim_img = torch.mm(fake_img_embedding, fake_img_embedding.T)       # (N, N)
    real_cos_sim_img = torch.mm(real_img_embedding, real_img_embedding.T)       # (N, N)
    fake_cos_sim_text = torch.mm(fake_text_embedding, fake_text_embedding.T)    # (N, N)
    real_cos_sim_text = torch.mm(real_text_embedding, real_text_embedding.T)    # (N, N)

    # Apply softmax (rowise)
    img_loss = F.kl_div(F.log_softmax(fake_cos_sim_img, dim=-1), F.softmax(real_cos_sim_img, dim=-1), reduction='batchmean')
    text_loss = F.kl_div(F.log_softmax(fake_cos_sim_text, dim=-1), F.softmax(real_cos_sim_text, dim=-1), reduction='batchmean')
    return img_loss + text_loss

def mode_seeking_loss(noise, text_embedding, img_embedding):
    N = text_embedding.shape[0]
    # Normalize features
    text_embedding = F.normalize(text_embedding, p=2, dim=-1)
    img_embedding = F.normalize(img_embedding, p=2, dim=-1)

    # Compute cosine similarity matrices
    cos_sim_text = torch.mm(text_embedding, text_embedding.T)                 # (N, N)
    cos_sim_img = torch.mm(img_embedding, img_embedding.T)                    # (N, N)
    noise_dist = (noise.unsqueeze(0) - noise.unsqueeze(1)).norm(dim=-1, p=2)  # (N, N)
    noise_dist[torch.eye(noise_dist.size(0)).bool()] = 1e6

    # Compute loss
    loss = ((cos_sim_text + cos_sim_img)/noise_dist).sum() / (N**2 - N)
    return loss

@torch.no_grad()
def evaluate_cos_similarities(img_embedding, text_embedding):
    img_embedding = F.normalize(img_embedding, p=2, dim=-1)
    text_embedding = F.normalize(text_embedding, p=2, dim=-1)
    
    # Cosine similarity between text embeddings
    cos_sim_text = torch.mm(text_embedding, text_embedding.T).cpu()
    cos_sim_img = torch.mm(img_embedding, img_embedding.T).cpu()

    cos_sim_text = (cos_sim_text.sum() / text_embedding.size(0) - 1) / (text_embedding.size(0) - 1) 
    cos_sim_img = (cos_sim_img.sum() / img_embedding.size(0) - 1) / (img_embedding.size(0) - 1)

    return cos_sim_img.cpu().numpy(), cos_sim_text.cpu().numpy()

