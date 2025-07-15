import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config


class GPT(GPT2Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
            self,
            inputs_embeds=None,
            attention_mask=None,
            **kwargs,
    ):
        output = super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
        return output.last_hidden_state


class DecisionTransformer(nn.Module):
    def __init__(self, env, action_dim, state_dim, device, model_name="gpt2"):
        super().__init__()

        self.device = device
        self.env = env

        embedding_dim = 768  # gpt2 hidden size
        self.embedding_dim = embedding_dim

        self.state_encoder = LinearEncoder(state_dim, embedding_dim)
        self.action_encoder = LinearEncoder(action_dim, embedding_dim)
        self.rtg_encoder = LinearEncoder(1, embedding_dim)
        self.timestep_encoder = LinearEncoder(1, embedding_dim)
        
        # config = GPT2Config.from_pretrained(model_name)
        # self.transformer = GPT(config)
        # self.transformer.load_state_dict(GPT2Model.from_pretrained(model_name).state_dict())
        self.transformer = GPT.from_pretrained(model_name)

        self.return_head = LinearEncoder(embedding_dim, 1)
        self.state_head = LinearEncoder(embedding_dim, state_dim)
        self.action_head = LinearEncoder(embedding_dim, action_dim)

    def _embed_sequence(self, states, actions, rtgs, timesteps):
        # timestep_embs = self.timestep_encoder(timesteps.unsqueeze(-1))  # (seq_len, encoding_dim)
        state_embs = self.state_encoder(states)
        action_embs = self.action_encoder(actions)
        rtg_embs = self.rtg_encoder(rtgs.unsqueeze(-1))

        state_embs = state_embs + timesteps  # (seq_len, encoding_dim)
        action_embs = action_embs + timesteps  # (seq_len, encoding_dim)
        rtg_embs = rtg_embs + timesteps  # (seq_len, encoding_dim)

        # sequence = torch.stack([rtg_embs, state_embs, action_embs, timestep_embs], dim=1).view(-1, self.encoding_dim)
        return state_embs, action_embs, rtg_embs

    def forward(self, states, actions, rtgs, timesteps, attention_mask=None):
        batch_size, seq_len = states.size(0), states.size(1)

        state_embs, action_embs, rtg_embs = self._embed_sequence(states, actions, rtgs, timesteps)
        inputs_embeds = torch.cat([rtg_embs, state_embs, action_embs], dim=1)
        output = self.transformer(inputs_embeds=inputs_embeds)

        output = output.reshape(batch_size, seq_len, 3, self.hidden_size).permute(0, 2, 1, 3) # check this

        return_preds = self.return_head(output[:,2]) # predict given state and action
        state_preds = self.state_head(output[:,2]) # predict given state and action
        action_preds = self.action_head(output[:,1]) # predict given state

        return return_preds, state_preds, action_preds

        seq_len = states.size(0)
        action_positions = torch.arange(2, 3*seq_len, 3, device=self.device)
        action_hidden = output[action_positions]

        action_logits = self.action_head(action_hidden)
        return action_logits
    
    def get_action(self, states, actions, rtgs, timesteps, **kwargs):
        
        _, action_preds, _ = self.forward(states, actions, rtgs, timesteps)
        return action_preds



class CNNEncoder(nn.Module):
    """
    CNN encoder for tokenizing the image states.
    """
    def __init__(self, width, height, output_dim, latent_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((width // 8) * (height // 8) * 64, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, output_dim),
        )

        self._init_parameters()

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class LinearEncoder(nn.Module):
    """
    Linear encoder for tokenizing the actions and rewards.
    """
    def __init__(self, input_dim, output_dim, latent_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            #nn.Linear(latent_dim, latent_dim),
            #nn.ReLU(),
            nn.Linear(latent_dim, output_dim),
        )

        self._init_parameters()

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
