import math
from collections import defaultdict
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
class AdamNSCL(Optimizer):
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

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, svd=False, thres=0.99,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, svd=svd,
                        thres=thres)
        super(AdamNSCL, self).__init__(params, defaults)

        self.eigens = defaultdict(dict)
        self.transforms = defaultdict(dict)

    def __setstate__(self, state):
        super(AdamNSCL, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
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
                            0), -1), self.transforms[n]).view_as(update)
                       
                    else:
                        update_ = torch.mm(update, self.transforms[n])
                    
                    # print("Null space updated..")

                else:
                    update_ = update
                p.data.add_(update_)
        return loss

    def adaptive_threshold(self, svals: torch.Tensor, offset: float = 0):
        points: np.ndarray = svals.cpu().numpy()
        assert points.ndim == 1
        if len(points) >= 128:
            fil_points = scipy.ndimage.gaussian_filter1d(points, sigma=10)
            _delta = 1
            diff_o1 = fil_points[:-_delta] - fil_points[_delta:]
            diff_o2 = diff_o1[:-1] - diff_o1[1:]
            _drop_ratio = 0.03
            drop_num = int(len(points) * _drop_ratio / 2)
            assert len(points) - drop_num >= 10
            valid_o2 = diff_o2[drop_num:-drop_num]
            thres_val = points[np.argmax(valid_o2) + int((len(points) - len(valid_o2)) / 2)]
        else:
            diff_o1 = points[:-1] - points[1:]
            diff_o2 = diff_o1[:-1] - diff_o1[1:]
            thres_val = points[np.argmax(diff_o2) + int((len(points) - len(diff_o2)) / 2)]
        i_thres = np.arange(len(points))[points >= thres_val].max()
        # assert 0 <= offset < 1, offset
        if -1 < offset < 1:
            i_thres = min(i_thres + int(offset * (len(points) - i_thres)), len(points) - 1)
            i_thres = max(0, i_thres)
        else:
            i_thres = max(min(i_thres + int(offset), len(points) - 1), 0)

        zero_idx = np.zeros(len(points), dtype=np.int64)
        zero_idx[i_thres:] = 1
        zero_idx = torch.as_tensor(torch.from_numpy(zero_idx), dtype=torch.bool, device=svals.device)
        return zero_idx


    def plot_sval_figures(self, svals_dict, distinguisher=None, offset = 0.0):
        plt.close()
        
        svals_dict_ary = {k: v['eigen_value'].cpu().numpy() for k, v in svals_dict.items()}
        
        
        fig, ax_list = plt.subplots(len(svals_dict_ary.keys())//4+1, 4)
        fig.set_figheight(60)
        fig.set_figwidth(15)
        for i, k in enumerate(svals_dict_ary.keys()):
            zero_idx = self.adaptive_threshold(svals_dict[k]['eigen_value'], offset=offset).cpu().numpy()
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
                    continue
                # print(n)
                ind = self.adaptive_threshold(self.eigens[n]['eigen_value'], offset=offset)
                # if (self.eigens[n]['eigen_value']==0).sum() != 0:
                print_log('reserving basis {}/{}; cond: {}, radio:{}'.format(
                    ind.sum(), self.eigens[n]['eigen_value'].shape[0],
                    self.eigens[n]['eigen_value'][0] /
                    self.eigens[n]['eigen_value'][ind][0],
                    self.eigens[n]['eigen_value'][ind].sum(
                    ) / self.eigens[n]['eigen_value'].sum()
                ))
                # GVV^T
                # get the columns
                basis = self.eigens[n]['eigen_vector'][:, ind]
                transform = torch.mm(basis, basis.transpose(1, 0))
                self.transforms[n] = transform / torch.norm(transform)
                self.transforms[n].detach_()
            # exit()

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
        if distinguisher != None:
            self.plot_sval_figures(self.eigens, distinguisher)

    def get_update(self, group, grad, p):
        amsgrad = group['amsgrad']
        state = self.state[p]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p.data)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1

        if group['weight_decay'] != 0:
            grad.add_(group['weight_decay'], p.data)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = max_exp_avg_sq.sqrt().add_(group['eps'])
        else:
            denom = exp_avg_sq.sqrt().add_(group['eps'])

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * \
            math.sqrt(bias_correction2) / bias_correction1
        update = - step_size * exp_avg / denom
        return update
