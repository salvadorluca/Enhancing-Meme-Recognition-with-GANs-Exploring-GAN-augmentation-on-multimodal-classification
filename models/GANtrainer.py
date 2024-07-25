from typing import Callable, List, Any, Dict
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

class GANTrainer():
    def __init__(self,
                 gen: Any,
                 disc: Any,
                 gen_optimizer: Any,
                 disc_optimizer: Any,
                 gen_scheduler: Any,
                 disc_scheduler: Any,
                 noise_dim: int,
                 gen_GAN_criterion: Callable,
                 disc_GAN_criterion: Callable,
                 num_gen_steps: int,
                 num_disc_steps: int,
                 metrics: Dict[str, Callable],
                 weight_clip: float = None,
                 lambda_gp: float = 0,
                 lambda_L1_gen: float = 0,
                 lambda_L2_gen: float = 0,
                 lambda_L1_disc: float = 0,
                 lambda_L2_disc: float = 0,
                 lambda_consistency: float = 0,
                 lambda_ms: float = 0,
                 device: Any = None,
                ):
        
        # Models
        self.gen = gen
        self.disc = disc
        self.noise_dim = noise_dim
        self.gen_params = [p for p in gen.parameters() if p.requires_grad]
        self.disc_params = [p for p in disc.parameters() if p.requires_grad]

        # Optimizers
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_scheduler = gen_scheduler
        self.disc_scheduler = disc_scheduler
        self.device = device

        # Loss functions and WGAN training
        self.gen_GAN_criterion = gen_GAN_criterion
        self.disc_GAN_criterion = disc_GAN_criterion
        self.weight_clip = weight_clip
        self.lambda_gp = lambda_gp
        self.num_gen_steps = num_gen_steps
        self.num_disc_steps = num_disc_steps

        # Regularization
        self.lambda_L1_gen = lambda_L1_gen
        self.lambda_L2_gen = lambda_L2_gen
        self.lambda_L1_disc = lambda_L1_disc
        self.lambda_L2_disc = lambda_L2_disc

        # 'Diversity' loss weights
        self.lambda_consistency = lambda_consistency
        self.lambda_ms = lambda_ms

        # Logs and metrics
        self.metrics = metrics
        self.metrics_log = {k: [] for k in metrics.keys()}
        # Move metrics to device
        for k in self.metrics.keys():
            self.metrics[k] = self.metrics[k].to(device)
        self.gen_loss_log = []
        self.disc_loss_log = []
        self.metrics_log = {k: [] for k in metrics.keys()}
        self.cos_sim_img_log = []
        self.cos_sim_text_log = []

    def gen_step(self, noise, real_img, real_text):
        self.gen_optimizer.zero_grad()
        fake_img, fake_text = self.gen(noise)
        fake_predictions = self.disc(fake_img, fake_text)
        gen_loss = self.gen_GAN_criterion(fake_predictions)

        # L1 and L2 regularization
        if self.lambda_L1_gen:
            gen_loss += self.lambda_L1_gen * L1_regularization(self.gen_params)

        if self.lambda_L2_gen:
            gen_loss += self.lambda_L2_gen * L2_regularization(self.gen_params)
        
        # Consistency loss
        if self.lambda_consistency:
            gen_loss += self.lambda_consistency * relation_consistency_loss(fake_img, fake_text, real_img, real_text)
        
        # Mode seeking loss (modified so it is based on cosine-similarity)
        if self.lambda_ms:
            gen_loss += self.lambda_ms * mode_seeking_loss(noise, fake_text, fake_img)
        
        gen_loss.backward()
        self.gen_optimizer.step()

        return gen_loss.item()
    
    def disc_step(self, fake_img, fake_text, real_img, real_text):
        self.disc_optimizer.zero_grad()
        real_predictions = self.disc(real_img, real_text)
        fake_predictions = self.disc(fake_img, fake_text)
        disc_loss = self.disc_GAN_criterion(fake_predictions, real_predictions)

        # Gradient penalty
        if self.lambda_gp:
            gradient_penalty = compute_gradient_penalty(self.disc, real_img, real_text, fake_img, fake_text, self.device)
            disc_loss += self.lambda_gp * gradient_penalty

        # L1 and L2 regularization
        if self.lambda_L1_disc:
            disc_loss += self.lambda_L1_disc * L1_regularization(self.disc_params)

        if self.lambda_L2_disc:
            disc_loss += self.lambda_L2_disc * L2_regularization(self.disc_params)
        
        disc_loss.backward()
        self.disc_optimizer.step()
        if self.weight_clip is not None:
            clamp_weights(self.disc, self.weight_clip)

        return disc_loss.item(), fake_predictions, real_predictions
    
    def train_epoch(self, dataloader):
        # Train models
        self.gen.train()
        self.disc.train()
        for i, (img_feature, text_feature, _) in enumerate(tqdm(dataloader, leave=True, desc='Train', colour='blue')):
            # Move data to device
            img_feature = img_feature.to(self.device)
            text_feature = text_feature.to(self.device)

            # Generate noise
            noise = torch.randn(img_feature.size(0), self.noise_dim).to(self.device)

            # Train generator
            set_model_require_grad(self.disc, False)
            set_model_require_grad(self.gen, True)
            for _ in range(self.num_gen_steps):
                gen_loss = self.gen_step(noise, img_feature, text_feature)
            self.gen_loss_log.append(gen_loss)

            # Train discriminator
            set_model_require_grad(self.disc, True)
            set_model_require_grad(self.gen, False)
            fake_img, fake_text = self.gen(noise)
            for _ in range(self.num_disc_steps):
                disc_loss, fake_predictions, real_predictions = self.disc_step(fake_img, fake_text, img_feature, text_feature)
            self.disc_loss_log.append(disc_loss)
            
            # Update logs
            self.gen_loss_log.append(gen_loss)
            self.disc_loss_log.append(disc_loss)
            self.metrics['acc_fake'].update(F.sigmoid(fake_predictions), torch.zeros_like(fake_predictions))
            self.metrics['acc_real'].update(F.sigmoid(real_predictions), torch.ones_like(real_predictions))
        for k in ['acc_fake', 'acc_real']:
            self.metrics_log[k].append(self.metrics[k].compute().cpu().numpy())
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        # Evaluate generator and discriminator
        self.gen.eval()
        self.disc.eval()

        # Reset metrics
        for k in self.metrics.keys():
            self.metrics[k].reset()
        
        text_sim_count = 0
        img_sim_count = 0
        for i, (img_feature, text_feature, _) in enumerate(tqdm(dataloader, leave=True, desc='Eval', colour='green')):
            # Move data to device
            img_feature = img_feature.to(self.device)
            text_feature = text_feature.to(self.device)

            # Generate noise
            noise = torch.randn(img_feature.size(0), self.noise_dim).to(self.device)

            # Generate fake samples
            fake_img, fake_text = self.gen(noise)

            # Compute predictions
            # fake_predictions = self.disc(fake_img, fake_text)
            # real_predictions = self.disc(img_feature, text_feature)

            # Update metrics
            # self.metrics['acc_fake'].update(F.sigmoid(fake_predictions), torch.zeros_like(fake_predictions))
            # self.metrics['acc_real'].update(F.sigmoid(real_predictions), torch.ones_like(real_predictions))
            
            # Compute cosine similarity
            img_sim, text_sim = evaluate_cos_similarities(fake_img, fake_text)
            img_sim_count += img_sim
            text_sim_count += text_sim
        
        # Update logs
        # for k in self.metrics.keys():
        #     self.metrics_log[k].append(self.metrics[k].compute().cpu().item())
        self.cos_sim_img_log.append(img_sim_count / len(dataloader))
        self.cos_sim_text_log.append(text_sim_count / len(dataloader))    

    def train(self, dataloader, num_epochs, fig, dh):
        for epoch in range(num_epochs):
            self.train_epoch(dataloader)
            self.evaluate(dataloader)

            # Update learning rate
            self.gen_scheduler.step()
            self.disc_scheduler.step()

            gen_epoch_loss = np.mean(self.gen_loss_log[-len(dataloader):])
            disc_epoch_loss = np.mean(self.disc_loss_log[-len(dataloader):])
            acc_real = self.metrics_log['acc_real'][-1]
            acc_fake = self.metrics_log['acc_fake'][-1]
            cos_sim_img = self.cos_sim_img_log[-1]
            cos_sim_text = self.cos_sim_text_log[-1]

            print(f'Epoch {epoch+1}/{num_epochs} - Gen Loss: {gen_epoch_loss:.4f} - Disc Loss: {disc_epoch_loss:.4f} - Acc Real: {acc_real:.4f} - Acc Fake: {acc_fake:.4f} - Cos Sim Img: {cos_sim_img:.4f} - Cos Sim Text: {cos_sim_text:.4f}')
            self.plot(fig, dh)

    def plot(self, fig, dh):
        ax = fig.axes
        ax[0].clear()
        ax[0].plot(self.gen_loss_log, label='Generator loss', color='orange')
        ax[0].plot(self.disc_loss_log, label='Discriminator loss', color='blue')
        # Set y-limits to include last 100 iterations
        ax[0].set_ylim([min(min(self.gen_loss_log[-1000:]), min(self.disc_loss_log[-1000:])), max(max(self.gen_loss_log[-1000:]), max(self.disc_loss_log[-1000:]))])
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('Training loss')
        ax[0].legend()
        ax[0].set_title('Training loss')

        # Print all metrics
        ax[1].clear()
        for key, value in self.metrics_log.items():
            ax[1].plot(value, label=key)
        ax[1].plot(self.cos_sim_img_log, label='Cosine similarity img', color='green')
        ax[1].plot(self.cos_sim_text_log, label='Cosine similarity text', color='red')
        ax[1].set_ylim([0, 1])
        ax[1].set_xlabel('Epoch')
        ax[1].legend()
        ax[1].set_title('Metrics')
        dh.update(fig)

    def save(self, path):
        torch.save({
            'gen': self.gen.state_dict(),
            'disc': self.disc.state_dict(),
            'gen_optimizer': self.gen_optimizer.state_dict(),
            'disc_optimizer': self.disc_optimizer.state_dict(),
            'gen_loss_log': self.gen_loss_log,
            'disc_loss_log': self.disc_loss_log,
            'metrics_log': self.metrics_log,
            'cos_sim_img_log': self.cos_sim_img_log,
            'cos_sim_text_log': self.cos_sim_text_log
        }, path)