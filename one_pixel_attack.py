import numpy as np
import torch
from torch.autograd import Variable
import utilities as ut
from scipy.optimize import differential_evolution
from torch.nn import functional as F

"""
One Pixel Attack
    Paper: https://arxiv.org/abs/1710.08864
"""

class OnePixelAttack():

    def __init__(self,
                 model,
                 device,
                 ):
        '''
        :param classifier (nn.Module): Neural network with callable, implemented forward function -- must expect inputs of size 784
        :param device (str): either 'cuda' or 'cpu'
        '''

        # Initialize classifier and device
        self.model = model
        self.device = device
        self.img = None
        self.pred_orig = None

    def preprocess(self, img):
        img = img.astype(np.float32)
        img = img.reshape(784)
        return img

    def perturb(self, x):
        adv_img = self.img.copy()

        # calculate pixel locations and values
        pixs = np.array(x).astype(int)
        loc = (pixs[0::3], pixs[1::3])
        val = pixs[2::3] / 255
        adv_img[loc] = val

        return adv_img

    def optimize(self, x):
        adv_img = self.perturb(x)

        inp = Variable(torch.from_numpy(self.preprocess(adv_img)).float().unsqueeze(0)).to(self.device)
        out = self.model.classifier(inp)
        prob = ut.softmax(out.data.cpu().numpy()[0])

        return prob[self.pred_orig]

    def callback(self, x, convergence):
        global pred_adv, prob_adv
        adv_img = self.perturb(x)

        inp = Variable(torch.from_numpy(self.preprocess(adv_img)).float().unsqueeze(0)).to(self.device)
        out = self.model.classifier(inp)
        prob = ut.softmax(out.data.cpu().numpy()[0])

        pred_adv = np.argmax(prob)
        prob_adv = prob[pred_adv]
        if pred_adv != self.pred_orig and prob_adv >= 0.9:
            return True
        else:
            pass

    def pixel_attack(self, d, shape, pred_orig, img):
        self.pred_orig = pred_orig
        bounds = [(0, shape[0] - 1), (0, shape[1] - 1), (0, 255)] * d
        adv_img = img
        if d > 0:
            result = differential_evolution(self.optimize, bounds, maxiter=600, popsize=10,
                                            tol=1e-5, callback=self.callback)
            adv_img = self.perturb(result.x)
        inp = Variable(torch.from_numpy(self.preprocess(adv_img)).float().unsqueeze(0)).to(self.device)
        out = self.model.classifier(inp)
        prob = ut.softmax(out.data.cpu().numpy())

        # Compute KL Divergence between q(z | x, y) and prior N(z | mu=0, sigma=1)
        adv_y = F.softmax(out, dim=-1).float()
        qm, qv = self.model.encoder(inp, adv_y)
        kl_z_all = adv_y * ut.kl_normal(
            qm, qv, self.model.z_prior[0], self.model.z_prior[1])  # kl_z_all shape = [batch_size * y_dim]
        kl_z = torch.sum(kl_z_all)  # scalar
        fake_y = torch.topk(adv_y, 1, dim=-1)[1].item()
        prob_fake_y = prob[0][fake_y]

        return kl_z, fake_y, prob_fake_y, inp
