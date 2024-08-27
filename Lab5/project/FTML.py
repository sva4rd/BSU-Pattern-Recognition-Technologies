import torch
import time


class FTML(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.6, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.step_num = 0
        self.total_time = 0

    def step(self):
        start = time.time()
        for group in self.param_groups:
            # For different groups, we might want to use different lr, regularizer, ...
            for p in group['params']:
                if p.grad is None:
                    continue
                gt = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    # State initialization
                    state['step'] = 0
                    state['vt'] = torch.zeros_like(p.data)
                    state['dt'] = torch.zeros_like(p.data)
                    state['zt'] = torch.zeros_like(p.data)

                state['step'] += 1
                t = state['step']
                vt, dt, zt = state['vt'], state['dt'], state['zt']

                eta = group['lr']
                beta1, beta2 = group['betas']
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                if group['weight_decay'] != 0:
                    gt.add_(group['weight_decay'], p.data)

                vt.mul_(beta2).addcmul_(1 - beta2, gt, gt)
                dt = bias_correction1 / eta * (torch.sqrt(vt / bias_correction2) + group['eps'])
                sigmat = dt - beta1 * state['dt']
                zt.mul_(beta1).add_((1 - beta1) * gt - sigmat * p.data)

                p.data = -zt / dt
                state['dt'] = dt
        self.step_num += 1
        self.total_time += time.time() - start

    def step_(self):
        start = time.time()
        for group in self.param_groups:
            # For different groups, we might want to use different lr, regularizer, ...
            for p in group['params']:
                gt = p.grad.data
                if gt is None:
                    print('skip one layer')
                    continue

                state = self.state[p]
                if len(state) == 0:
                    # State initialization
                    state['step'] = 0
                    state['mt'] = torch.zeros_like(p.data)
                    state['vt'] = torch.zeros_like(p.data)
                    state['Qt'] = torch.zeros_like(p.data)
                    state['Ht'] = torch.zeros_like(p.data)
                    state['avg_Qtxt'] = torch.zeros_like(p.data)

                state['step'] += 1
                t = state['step']
                mt, vt, Qt, Ht = state['mt'], state['vt'], state['Qt'], state['Ht']
                avg_Qtxt = state['avg_Qtxt']

                eta = group['lr']
                beta1, beta2 = group['betas']
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                if group['weight_decay'] != 0:
                    gt.add_(group['weight_decay'], p.data)

                mt.mul_(beta1).add_(1 - beta1, gt)
                mt_corrected = mt / bias_correction1
                vt.mul_(beta2).addcmul_(1 - beta2, gt, gt)
                Hessian = (torch.sqrt(vt / bias_correction2) + group['eps']) / eta
                Qt = (bias_correction1 * Hessian - beta1 * (1 - beta1 ** (t - 1)) * Ht) / (1 - beta1)
                avg_Qtxt.mul_(beta1).addcmul_(1 - beta1, Qt, p.data)
                p.data = (avg_Qtxt - mt) / bias_correction1 / Hessian

                state['Ht'] = Hessian

        self.step_num += 1
        self.total_time += time.time() - start

    def sec_per_step(self):
        return self.total_time / self.step_num