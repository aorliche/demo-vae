
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression

def to_torch(x):
    return torch.from_numpy(x).float()

def to_cuda(x, use_cuda):
    if use_cuda:
        return x.cuda()
    else:
        return x

def to_numpy(x):
    return x.detach().cpu().numpy()

class VAE(nn.Model):
    def __init__(self, input_dim, latent_dim, demo_dim, use_cuda=True):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.demo_dim = demo_dim
        self.use_cuda = use_cuda
        self.enc1 = to_cuda(nn.Linear(input_dim, 1000).float(), use_cuda)
        self.enc2 = to_cuda(nn.Linear(1000, latent_dim).float(), use_cuda)
        self.dec1 = to_cuda(nn.Linear(latent_dim+demo_dim, 1000).float(), use_cuda)
        self.dec2 = to_cuda(nn.Linear(1000, latent_dim).float(), use_cuda)

    def enc(self, x):
        x = F.relu(self.enc1(x))
        z = self.enc2(x)
        return z

    def gen(self, n):
        return to_cuda(torch.randn(n, self.latent_dim).float(), self.use_cuda)

    def dec(self, z, demo):
        z = to_cuda(torch.cat([z, demo], dim=1), self.use_cuda)
        x = F.relu(self.dec1(z))
        x = self.dec2(x)
        return x

def rmse(a, b, mean=torch.mean):
    return mean((a-b)**2)**0.5

def latent_loss(z, use_cuda=True):
    C = z.T@z
    tgt1 = to_cuda(torch.eye(z.shape[-1]).float(), use_cuda)
    tgt2 = to_cuda(torch.zeros(z.shape[-1]).float(), use_cuda)
    loss_C = rmse(C, tgt1)
    loss_mu = rmse(mu, tgt2)
    return loss_C, loss_mu, C, mu

def decor_loss(z, demo, use_cuda=True):
    ps = []
    losses = []
    for d in demo:
        d = d - torch.mean(d)
        p = torch.einsum('n,nz->z', d, z)
        tgt = to_cuda(torch.zeros(z.shape[-1]).float(), use_cuda)
        loss = rmse(p, tgt)
        losses.append(loss)
        ps.append(p)
    losses = torch.stack(losses)
    return losses, ps

def pretty(x):
    return f'{round(float(x), 4)}'

def demo_to_torch(demo, demo_types, pred_stats):
    demo_t = []
    for d,t,s in zip(demo, demo_types, pred_stats):
        if t == 'continuous':
            demo_t.append(to_cuda(to_torch(d), vae.use_cuda))
        elif t == 'categorical':
            for ss in s:
                idx = (d == ss).astype('int')
                zeros = torch.zeros(len(d))
                zeros[idx] = 1
                demo_t.append(to_cuda(zeros, vae.use_cuda))
    demo_t = torch.stack(demo_t).permute(1,0)
    return demo_t

def train_vae(vae, x, demo, demo_types, nepochs, pperiod, bsize, loss_C_mult, loss_mu_mult, loss_rec_mult, loss_decor_mult, loss_pred_mult):
    # Get linear predictors for demographics
    pred_w = []
    pred_i = []
    # Pred stats are mean and std for continuous, and a list of all values for continuous
    pred_stats = []
    for i,d,t in zip(range(len(demo)), demo, demo_types):
        print(f'Fitting auxillary guidance model for demographic {i} {t}...', end='')
        if t == 'continuous':
            pred_stats.append([np.mean(d), np.std(d)])
            reg = Ridge(alpha=1).fit(x, d)
            reg_w = to_cuda(to_torch(reg.coef_), vae.use_cuda)
            reg_i = reg.intercept_
        elif t == 'categorical':
            pred_stats.append(list(set(list(d))).sort())
            reg = LogisticRegression(C=1).fit(x, d)
            for i in range(len(reg.coef_)):
                reg_w = to_cuda(to_torch(reg.coef_[i]), vae.use_cuda)
                reg_i = reg.intercept_[i]
        else:
            print(f'demographic type "{t}" not "continuous" or "categorical"')
            raise Exception('Bad demographic type')
        pred_w.push(reg_w)
        pred_i.push(reg.intercept_)
        print(' done')
    # Convert input to pytorch
    print('Converting input to pytorch')
    x = to_cuda(to_torch(x), vae.use_cuda)
    # Convert demographics to pytorch
    print('Converting demographics to pytorch')
    demo_t = demo_to_torch(demo, demo_types, pred_stats)
    # Training loop
    print('Beginning VAE training')
    ce = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(vae.parameters(), lr=1e-4, weight_decay=0)
    for e in range(nepochs):
        for bs in range(0,len(x),bsize):
            xb = x[bs:(bs+bsize)]
            db = demo_t[bs:(bs+bsize)]
            optim.zero_grad()
            # Reconstruct
            z = vae.enc(xb)
            y = vae.dec(z, db)
            loss_C, loss_mu, _, _ = latent_loss(z)
            loss_decor = decor_loss(z, db)
            loss_rec = rmse(xb, y)
            loss_C = loss_C*loss_C_mult
            loss_mu = loss_mu*loss_mu_mult
            loss_decor = loss_decor*loss_decor_mult
            loss_rec = loss_rec*loss_rec_mult
            # Sample demographics
            demo_gen = []
            for s,t in zip(pred_stats, demo_types):
                if t == 'continuous':
                    mu = s[0]
                    std = s[1]
                    dd = torch.ones(100).float()
                    dd = dd*std+mu
                elif t == 'categorical':
                    idx = random.randint(0, len(s)-1)
                    for i in range(s):
                        if idx == i:
                            dd = torch.ones(100).float()
                        else:
                            dd = torch.zeros(100).float()
                dd = to_cuda(dd, vae.use_cuda)
                demo_gen.append(dd)
            demo_gen = torch.stack(demo_gen).permute(1,0)
            # Generate
            z = vae.gen(100)
            y = vae.dec(z, demo_gen)
            # Regressor/classifier guidance loss
            losses_pred = []
            dg_idx = 0
            for t in zip(demo_types):
                if t == 'continuous':
                    yy = y@pred_w[dg_idx]+pred_i[dg_idx]
                    loss = loss_pred_mult*rmse(demo_gen[:,dg_idx], yy)
                    losses_pred.append(loss)
                    dg_idx += 1
                elif t == 'categorical':
                    loss = 0
                    for i in range(s):
                        yy = y@pred_w[dg_idx]+pred_i[dg_idx]
                        loss += loss_pred_mult*ce(torch.stack([-yy, yy], dim=1), demo_gen[:,dg_idx].long())
                        dg_idx += 1
                    losses_pred.append(loss)
            total_loss = loss_C + loss_mu + loss_rec + loss_decor + sum(losses_pred)
            total_loss.backward()
            optim.step()
            if e%pperiod == 0 or e == nepochs-1:
                print(f'Epoch {e} ', end='')
                print(f'ReconLoss {pretty(loss_rec)} ', end='')
                print(f'CovarianceLoss {pretty(loss_C)} ', end='')
                print(f'MeanLoss {pretty(loss_mu)} ', end='')
                print(f'DecorLoss {pretty(loss_decor)} ', end='')
                losses_pred = [pretty(loss) for loss in losses_pred]
                print(f'GuidanceLosses {losses_pred} ', end='')
                print()
    print('Training complete.')
    return pred_stats
                

