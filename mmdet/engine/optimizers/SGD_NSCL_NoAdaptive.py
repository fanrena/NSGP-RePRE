import math
from collections import defaultdict
import scipy.ndimage
import torch
from torch.optim.optimizer import Optimizer
from mmengine.registry import OPTIMIZERS
from mmengine.logging import MessageHub, MMLogger, print_log
import scipy
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import os.path as osp

@OPTIMIZERS.register_module()
class SGDNSCLNA(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, momentum=0, dampening=0, nesterov=False, svd=False, thres=1.001,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, nesterov=nesterov,
                        weight_decay=weight_decay, svd=svd,
                        thres=thres)
        super(SGDNSCLNA, self).__init__(params, defaults)
        
        self.eigens = defaultdict(dict)
        self.transforms = defaultdict(dict)
        self.count = 0
        self.thres = thres

    def __setstate__(self, state):
        super(SGDNSCL, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('svd', False)
            group.setdefault('names', [])

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            svd = group['svd']
            for n, p in zip(group['names'], group['params']):
                # if p.grad is None:
                #     continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')

                update = self.get_update(group, grad, p)
                # print(svd, self.transforms)
                if svd and len(self.transforms) > 0 and n in self.transforms.keys():
                    if len(update.shape) == 4:
                        # the transpose of the manuscript
                        update_ = torch.mm(update.view(update.size(
                            0), -1), self.transforms[n])
                        update_ = update_.view_as(update)
                       
                    else:
                        update_ = torch.mm(update, self.transforms[n])
                    
                    # print("Null space updated..")
                    # if 'weight' in n and self.count % 500 == 0:
                    #     print(n)
                    #     print(update_.norm())
                    #     print(update.norm())
                    #     print((update - update_).norm())
                    #     if len(update.shape) == 4:
                    #         # the transpose of the manuscript
                    #         get_angle(update.view(update.size(
                    #             0), -1), update_.view(update.size(
                    #             0), -1))
                    #     else:
                    #         get_angle(update, self.transforms[n])

                    # if 'neck' in n:
                    #     update_ = update_ * 100
                    
                else:
                    update_ = update
                p.data.add_(update_)
        return loss



    def plot_sval_figures(self, svals_dict, distinguisher=None, offset = 0.0):
        plt.close()
        
        svals_dict_ary = {k: v['eigen_value'].cpu().numpy() for k, v in svals_dict.items()}
        
        
        fig, ax_list = plt.subplots(len(svals_dict_ary.keys())//4+1, 4)
        fig.set_figheight(60)
        fig.set_figwidth(15)
        for i, k in enumerate(svals_dict_ary.keys()):
            zero_idx = self.eigens[p]['eigen_value'] <= self.eigens[p]['eigen_value'][-1] * thres
            points: np.ndarray = svals_dict_ary[k]
            i_thres = np.arange(len(points))[zero_idx].min()

            ax_list[i // 4, i % 4].plot(np.arange(i_thres + 1), points[:i_thres + 1], color='blue')
            ax_list[i // 4, i % 4].plot(np.arange(i_thres, len(points)), points[i_thres:], color='red')
            ax_list[i // 4, i % 4].set_title(k)
        if not osp.exists(save_dir := osp.join('./', 'figures')):
            os.mkdir(save_dir)
        fig.tight_layout()
        fig.savefig(osp.join(save_dir, save_name := f"svals_task{1}_{distinguisher}.png"))
        plt.close()

    def get_transforms(self, offset=0.0):
        for group in self.param_groups:
            svd = group['svd']
            if svd is False:
                continue
            for n, p in zip(group['names'], group['params']):
                # if p.grad is None:
                #     continue
                if n not in self.eigens.keys():
                    print_log(f"missing keys: {n}")
                    continue
                # print(offset)
                # if 'neck' not in n:
                #     offset = 0
                # else:
                #     offset = -1
                # print(n, offset)
                thres = group['thres']
                ind = self.eigens[n]['eigen_value'] <= self.eigens[n]['eigen_value'][-1] * thres
                # if (self.eigens[n]['eigen_value']==0).sum() != 0:
                print_log('{}: reserving basis {}/{}; cond: {}, radio:{}'.format(
                    n,
                    ind.sum(), self.eigens[n]['eigen_value'].shape[0],
                    self.eigens[n]['eigen_value'][0] /
                    self.eigens[n]['eigen_value'][ind][0],
                    self.eigens[n]['eigen_value'][ind].sum(
                    ) / self.eigens[n]['eigen_value'].sum()
                ))
                # GVV^T
                # get the columns
                basis = self.eigens[n]['eigen_vector'][:, ind]
                # inverse_transform = torch.pinverse(basis)
                # self.inverse_transform[n] = inverse_transform / inverse_transform.transpose(1,0)
                transform = torch.mm(basis, basis.transpose(1, 0))
                if 'backbone' in n:
                    self.transforms[n] = transform / torch.norm(transform)
                else:
                    self.transforms[n] = transform
                # print(n, "transform - I", self.transforms[n] - torch.eye(transform.shape[0]).to(transform))
                self.transforms[n].detach_()

    def get_eigens(self, fea_in, distinguisher=None):
        for group in self.param_groups:
            svd = group['svd']
            if svd is False:
                continue
            for n, p in zip(group['names'], group["params"]):
                # if p.grad is None:
                #     continue
                if n not in fea_in.keys():
                    continue
                # print(n)
                eigen = self.eigens[n]
                # print(fea_in[n].keys())
                _, eigen_value, eigen_vector = torch.svd(fea_in[n], some=False)
                eigen['eigen_value'] = eigen_value
                # print(n, eigen_value)
                eigen['eigen_vector'] = eigen_vector
        # if distinguisher != None:
        #     self.plot_sval_figures(self.eigens, distinguisher)
            

    def get_update(self, group, grad, p):
        nesterov = group["nesterov"]
        state = self.state[p]

        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['previous_grad'] = torch.zeros_like(p.data)

        exp_avg = state['previous_grad']
        state['step'] += 1

        if group['weight_decay'] != 0:
            grad.add_(group['weight_decay'], p.data)
            
        if group['momentum'] != 0:
            if state['step'] > 1:
                exp_avg.mul_(group['momentum']).add_(1-group['dampening'], grad)
            else:
                exp_avg.add_(grad)
        # State initialization
            if group['nesterov']:
                grad.add_(group['momentum'], exp_avg)
            else:
                grad = exp_avg

        step_size = group['lr'] * grad
        update = - step_size
        return update



def get_angle(u, proj):
    '''
    u is the vector, v is the eigens
    '''
    # 计算 u 在基 v 上的投影
    # print(u.shape, v.shape)
    # proj = torch.mm(u, transform.to(u.device))
    # 计算投影的模长
    proj_norm = torch.norm(proj, dim=1)
    
    # 计算 u 的模长
    u_norm = torch.norm(u, dim=1)

    # 计算夹角的余弦值
    cos_theta = torch.diag(torch.mm(u, proj.t())) / (proj_norm * u_norm)
    # 计算夹角
    theta = torch.rad2deg(torch.acos(cos_theta))
    print(cos_theta)
    print(torch.acos(cos_theta))
    print(theta)

    return theta