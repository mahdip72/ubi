import os
import pickle
import torch
from csv import writer


def append_list_as_row(save_path, mode, list_of_elem):
    if os.path.exists(os.path.join(save_path, 'log.csv')):
        with open(os.path.join(save_path, 'log.csv'), 'a+', newline='') as file:
            # Create a writer object from csv module
            if mode == 'train':
                file.write(', '.join([str(i) for i in list_of_elem]) + ', ')
            elif mode == 'valid':
                file.write(', '.join([str(i) for i in list_of_elem[1:]]) + '\n')
    else:
        with open(os.path.join(save_path, 'log.csv'), 'a+', newline='') as file:
            csv_writer = writer(file)
            titles = ['epoch', 'loss', 'acc', 'precision', 'recall', 'f1',
                      'val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1']
            csv_writer.writerow(titles)


def save_tokenizer(tokenizer, path):
    # saving
    with open(os.path.join(path, 'tokenizer.pkl'), 'wb') as file:
        pickle.dump(tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_tokenizer(path):
    # loading
    with open(os.path.join(path, 'tokenizer.pkl'), 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def test_gpu_cuda():
    print('Testing gpu and cuda:')
    print('\tcuda is available:', torch.cuda.is_available())
    print('\tdevice count:', torch.cuda.device_count())
    print('\tcurrent device:', torch.cuda.current_device())
    print(f'\tdevice:', torch.cuda.device(0))
    print('\tdevice name:', torch.cuda.get_device_name(), end='\n\n')


def print_log(c, loss_value, acc, precision, recall, f1, steps):
    if c % steps == 0:
        print(f"step {c} - loss: {loss_value:>8f} acc: {acc :>0.2f}% "
              f"precision: {precision:>4f} recall: {recall:>4f} f1: {f1:>4f}\t")


def write_tensorboard(compute_loss, tools, write, e, mode):
    acc = round(tools['accuracy'].compute().item() * 100, 2)
    precision = round(tools['precision'].compute().item(), 4)
    recall = round(tools['recall'].compute().item(), 4)
    f1 = round(tools['f1'].compute().item(), 4)

    # writing tensorboard logs
    write.add_scalar('loss', compute_loss, e)
    write.add_scalar('acc', acc, e)
    write.add_scalar('precision', precision, e)
    write.add_scalar('recall', recall, e)
    write.add_scalar('f1', f1, e)

    list_of_elem = [e, compute_loss, acc, precision, recall, f1]
    append_list_as_row(tools['save_path'], mode=mode, list_of_elem=list_of_elem)


def write_test_results(compute_loss, tools, write):
    acc = round(tools['accuracy'].compute().item() * 100, 2)
    precision = round(tools['precision'].compute().item(), 4)
    recall = round(tools['recall'].compute().item(), 4)
    f1 = round(tools['f1'].compute().item(), 4)
    auc = round(tools['auc'].compute().item(), 4)
    specificity = round(tools['specificity'].compute().item(), 4)
    mcc = round(tools['mcc'].compute().item(), 4)
    cm = tools['matrix'].compute().cpu().numpy()

    # writing tensorboard logs
    write.add_scalar('loss', compute_loss, 0)
    write.add_scalar('acc', acc, 0)
    write.add_scalar('precision', precision, 0)
    write.add_scalar('recall', recall, 0)
    write.add_scalar('f1', f1, 0)

    list_of_elem = [compute_loss, acc, precision, recall, f1, specificity, auc, mcc]

    with open(os.path.join(tools['save_path'], tools['test_name']+'.csv'), 'w', newline='') as file:
        csv_writer = writer(file)
        titles = ['test_loss', 'test_acc', 'test_precision', 'test_recall', 'test_f1',
                  'test_specificity', 'test_auc', 'test_mcc']
        csv_writer.writerow(titles)
        file.write(', '.join([str(i) for i in list_of_elem]))

    with open(os.path.join(tools['save_path'], 'confusion_matrix.txt'), 'w', newline='') as file:
        file.write(str(cm))


def compute_sam(tools, loss, model, x, p, y, w):
    tools['scaler'].scale(loss).backward()
    tools['scaler'].unscale_(tools['optimizer'])
    torch.nn.utils.clip_grad_norm_(model.parameters(), tools['grad_clip'])
    tools['optimizer'].first_step(zero_grad=True)
    tools['scaler'].update()

    # 2nd pass
    with torch.cuda.amp.autocast(enabled=tools['mixed_precision']):
        pred_2 = model(x, p)
        loss_2 = tools['loss'](pred_2, y) * w
        loss_2 = torch.mean(loss_2)

    tools['scaler'].scale(loss_2).backward()
    tools['scaler'].unscale_(tools['optimizer'])
    torch.nn.utils.clip_grad_norm_(model.parameters(), tools['grad_clip'])
    tools['optimizer'].second_step(zero_grad=True)
    tools['scaler'].update()
