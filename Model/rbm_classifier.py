import torch
import tqdm
import torch.nn.functional as F
from Dataloader.dataset import LetterImgDataset

class RBM(torch.nn.Module):
    def __init__(self, num_visible, num_hidden, n_C, k=1, lr=0.001, use_cuda=False):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.lr = lr

        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.W = torch.randn(num_hidden, num_visible, dtype=torch.float32, device=self.device) * 0.1
        self.b_h = torch.rand(num_hidden, dtype=torch.float32, device=self.device) * -2
        self.b_v = torch.zeros(num_visible, dtype=torch.float32, device=self.device)

        self.n_C = n_C

        self.U = torch.randn(num_hidden, n_C, dtype=torch.float32, device=self.device) * 0.1
        self.b_y = torch.zeros(n_C, dtype=torch.float32, device=self.device)

    def random_probabilities(self, shape):
        random_p = torch.rand(shape, dtype=torch.float32, device=self.device)
        return random_p


    def sample_h(self, v):
        x, y = v
        x = x.float()
        y = y.float()
        if self.use_cuda:
            x = x.cuda()
            y = y.cuda()
        h = torch.sigmoid(torch.mm(x, self.W.t()) + self.b_h + torch.mm(y, self.U.t())) # n_sample * n_hidden
        h_select = h >= self.random_probabilities(h.shape)
        return h, h_select.float()

    def sample_v(self, h):
        x_prob = torch.sigmoid(torch.matmul(h, self.W) + self.b_v)
        x_sample = x_prob >= self.random_probabilities(x_prob.shape) # n_sample * n_visible

        y_dist = self.b_y + torch.mm(h, self.U) # n_sample * n_C
        y_prob = torch.softmax(y_dist, dim=1)
        y_select = torch.multinomial(y_prob, num_samples=1).squeeze(1)
        y_sample = LetterImgDataset.one_hot(self.n_C, y_select)

        return x_prob, y_prob, (x_sample.float(), y_sample.float())
    
    def contrastive_divergence(self, v0):
        x, y = v0
        if self.use_cuda:
            x = x.cuda()
            y = y.cuda()
        num_samples = len(x)

        h_sample, h_select = self.sample_h(v0)
        hk_prob, hk = h_sample, h_select
        vk = v0

        with torch.no_grad():
            for _ in range(self.k):
                _, _, vk = self.sample_v(hk)
                hk_prob, hk = self.sample_h(vk)

        xk, yk = vk

        positive_grad_x = torch.matmul(h_sample.t(), x)
        negative_grad_x = torch.matmul(hk_prob.t(), xk)

        positive_grad_y = torch.matmul(h_sample.t(), y)
        negative_grad_y = torch.matmul(hk_prob.t(), yk)

        self.W_grad = (positive_grad_x - negative_grad_x) / num_samples
        self.U_grad = (positive_grad_y - negative_grad_y) / num_samples
        self.b_h_grad = (h_sample.mean(0) - hk_prob.mean(0))
        self.b_v_grad = (x.mean(0) - xk.mean(0))
        self.b_y_grad = (y.mean(0) - yk.mean(0))

        self.update_weights(lr=self.lr)


    def update_weights(self, lr=0.001):
        self.W += lr * self.W_grad
        self.U += lr * self.U_grad
        self.b_h += lr * self.b_h_grad
        self.b_v += lr * self.b_v_grad
        self.b_y += lr * self.b_y_grad


    def fit(self, data, epochs=10, val_data=None, val_frequency=0.1):
        val_list = None
        if val_data is not None:
            val_list = list(iter(val_data))
        val_ctr = 0
        n_batches, batches, val_losses = 0, [], []

        for epoch in range(epochs):
            epoch_progress = tqdm.tqdm(data, desc=f"Epoch {epoch + 1}/{epochs}", leave=False, bar_format="{l_bar}{r_bar}")
            
            val_every = int(len(data) * val_frequency)
            for i, batch in enumerate(epoch_progress):
                self.contrastive_divergence(batch)

                if val_list is not None and i % val_every == 0:
                    val_batch = val_list[val_ctr]
                    val_ctr = (val_ctr + 1) % len(val_list)
                    val_loss = self.calc_loss(val_batch)
                    epoch_progress.set_postfix({"val_loss": val_loss.item()})
                    val_losses.append(val_loss.item())
                    batches.append(n_batches)
            n_batches += 1

        return val_losses, batches

    def predict_y(self, x_data):

        """
        Following code is not stable, but it represents the original formula from the paper.
        we utilize a softmax for stability along with claculating log of the product.

        p = torch.mm(x_data, self.W.t()) + self.b_h

        combined = p.unsqueeze(2) + self.U.unsqueeze(0) # n_sample * n_hidden * n_C

        combined = 1 + torch.exp(combined)

        prod = torch.prod(combined, dim=1) # n_sample * n_C

        y_dist = torch.exp(self.b_y).unsqueeze(0) * prod # n_sample * n_C

        y_prob = y_dist / torch.sum(y_dist, dim=1, keepdim=True) # n_sample * n_C

        y_select = torch.multinomial(y_prob, num_samples=1).squeeze(1)

        return y_prob, y_select
        """

        if self.use_cuda:
            x_data = x_data.cuda()

        p = torch.mm(x_data, self.W.t()) + self.b_h

        combined = p.unsqueeze(2) + self.U.unsqueeze(0) # n_sample * n_hidden * n_C
        combined = torch.log(1 + torch.exp(combined)) 

        sum = torch.sum(combined, dim=1)

        y_dist = self.b_y.unsqueeze(0) + sum

        y_prob = torch.softmax(y_dist, dim=1) # n_sample * n_C
        y_select = torch.argmax(y_prob, dim=1)

        return y_prob, y_select
    
    def energy_function(self, v, h_select=None):
        x, y = v
        if self.use_cuda:
            x = x.cuda()
            y = y.cuda()
        if h_select is None:
            _, h_select = self.sample_h(v)
        
        energy = torch.zeros(len(x), dtype=torch.float32, device=self.device)   
        # print("here")
        energy -= torch.sum(torch.mm(h_select, self.W) * x, dim=1)
        # print("here1")
        energy -= torch.sum(x * self.b_v, dim=1)
        # print("here2")
        energy -= torch.sum(h_select * self.b_h, dim=1)
        # print("here3")
        energy -= torch.sum(y * self.b_y, dim=1)
        # print("here4")
        energy -= torch.sum(torch.mm(h_select, self.U) * y, dim=1)
        
        return energy
    
    def free_energy(self, v):
        x, y = v
        if self.use_cuda:
            x = x.cuda()
            y = y.cuda()

        free_energy = torch.zeros(len(x), dtype=torch.float32, device=self.device)
        free_energy -= torch.sum(x * self.b_v, dim=1)
        free_energy -= torch.sum(y * self.b_y, dim=1)
        free_energy -= torch.sum(
            torch.log(
                1 + torch.exp(
                    torch.mm(x, self.W.t()) + self.b_h + torch.mm(y, self.U.t())
                )
            ), dim=1)
        
        return free_energy
    
    def calc_loss(self, v):
        x, y = v
        if self.use_cuda:
            x = x.cuda()
            y = y.cuda()
        h_sample, h_select = self.sample_h(v)
        
        hk = h_select.clone()
        vk = v

        for _ in range(self.k):
            _, _, vk = self.sample_v(hk)
            _, hk = self.sample_h(vk)

        xk, yk = vk

        f_data = self.free_energy(v)
        f_k = self.free_energy(vk)

        return torch.mean(f_data - f_k)