import torch
import tqdm

class RBM(torch.nn.Module):
    def __init__(self, num_visible, num_hidden, k=1):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k

        self.W = torch.randn(num_hidden, num_visible) * 0.1
        self.b_h = torch.zeros(num_hidden)
        self.b_v = torch.zeros(num_visible)

    def sample_h(self, v):
        h_prob = torch.sigmoid(torch.matmul(v, self.W.t()) + self.b_h)
        h_sample = torch.bernoulli(h_prob)
        return h_sample, h_prob

    def sample_v(self, h):
        v_prob = torch.sigmoid(torch.matmul(h, self.W) + self.b_v)
        v_sample = torch.bernoulli(v_prob)
        return v_sample, v_prob

    def contrastive_divergence(self, v0):
        h0, _ = self.sample_h(v0)
        vk = v0.clone()
        hk = h0.clone()

        for _ in range(self.k):
            vk, _ = self.sample_v(hk)
            hk, _ = self.sample_h(vk)

        positive_grad = torch.matmul(h0.t(), v0)
        negative_grad = torch.matmul(hk.t(), vk)

        self.W += (positive_grad - negative_grad) / v0.size(0)
        self.b_h += (h0.mean(0) - hk.mean(0))
        self.b_v += (v0.mean(0) - vk.mean(0))

    def fit(self, data, epochs=10): 
        for epoch in range(epochs):
            for batch in tqdm.tqdm(data, desc=f"Epoch {epoch + 1}/{epochs}", leave=False, bar_format="{l_bar}{r_bar}"):
                self.contrastive_divergence(batch)


    def transform(self, data):
        _, h_prob = self.sample_h(data)
        return h_prob
    

    