# MIT License
#
# Original work Copyright (c) 2021 THUML @ Tsinghua University
# Modified work Copyright (c) 2025 DACElab @ University of Toronto
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, visual_all, CheckConvergence
from utils.metrics import metric, Evaluator
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

loss_fn = nn.MSELoss(reduction="mean")

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        if not hasattr(args, 'weight_decay'):
            args.weight_decay = 1e-4
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.model_optim = self._select_optimizer()
        if self.args.model == 'KRNO':
            self._setup_qudrature()

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.optim == 'AdamW':
            model_optim = torch.optim.AdamW(self.model.parameters(),
                                         lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        elif self.args.optim == 'Adam':
            model_optim = optim.Adam(self.model.parameters(),
                                  lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        # model_optim = optim.Adam(self.model.parameters(),
        #                           lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        
        if self.args.model == 'KRNO':
            criterion = self.model.loss_fn
        else:
            criterion = nn.MSELoss()
        return criterion
    
    def _setup_qudrature(self):
        # setting up computational grid for training
        input_res = self.args.seq_len #24
        # quad_grid_in = self.model.quadrature.get_quad_grid(
        #     self.model.quadrature.trapezoidal_vecs, [input_res], [0], [1]
        # )
        quad_grid_in = self.model.quadrature.get_quad_grid(
            self.model.quadrature.midpoint_vecs, [input_res], [0], [1]
        )
        if self.args.seq_len == self.args.pred_len:
            cart_grid    = self.model.quadrature.quad_to_cartesian_grid(quad_grid_in)
            self.grid_in = self.model.quadrature.cart_grid_to_device(cart_grid, self.device)
            self.grid_out = self.grid_in
        else:
            self.grid_in = self.model.quadrature.quad_grid_to_device(quad_grid_in, self.device)
            out_res = self.args.pred_len
            out_ub  = out_res/input_res
            quad_grid_out = self.model.quadrature.get_quad_grid(
                        self.model.quadrature.midpoint_vecs, [out_res], [0], [out_ub]
                    )
            self.grid_out = self.model.quadrature.quad_grid_to_device(quad_grid_out, self.device)

        if self.args.data in ['m4', 'Cryptos', 'Traj']:
            out_res = self.args.test_output_length
            out_ub  = out_res/input_res
            quad_grid_out = self.model.quadrature.get_quad_grid(
                        self.model.quadrature.midpoint_vecs, [out_res], [0], [out_ub]
                    )
            self.test_grid_out = self.model.quadrature.quad_grid_to_device(quad_grid_out, self.device)




    def vali(self, vali_loader, criterion, setting=None, flag='train', onestep_pred=False, test=0):

        if test:
            print('loading model')
            checkpoint = torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            self.model.load_state_dict(checkpoint['model_state_dict'])

        eval_loss = []
        all_preds = []
        all_trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # batch_y_mark = batch_y_mark - batch_x_mark[:,0].reshape(-1,1) 
                # batch_x_mark = batch_x_mark - batch_x_mark[:,0].reshape(-1,1) 

                # # normailizing output times to [0, 1] 
                # batch_y_mark = batch_y_mark - batch_x_mark[:,-1].reshape(-1,1) 
                # batch_y_mark = batch_y_mark/batch_y_mark[:, -1].reshape(-1,1) 

                # normailizing input times to [0, 1]
                batch_x_mark = batch_x_mark - batch_x_mark[:,0].reshape(-1,1)
                batch_x_mark = batch_x_mark/batch_x_mark[:, -1].reshape(-1,1) 

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Extract shape of the input tensor for reshaping (i.e. stacking the
                # time and channels dimension together)
                inp_shape = list(batch_x.shape)
                inp_shape = inp_shape[:-1]
                inp_shape.append(-1)
                batch_x_mark = batch_x_mark.reshape(inp_shape)
                # batch_y_mark = batch_y_mark.reshape(inp_shape)

                # Reshape input tensor into [b, input seq length , channels]
                # inp_mark = torch.cat([batch_x_mark, batch_y_mark], dim=2).to(self.device)
                inp_mark = batch_x_mark

                # pdb.set_trace()
                # Model run
                if self.args.model == 'KRNO':
                    if self.args.seq_len == self.args.pred_len:
                        inp_mark = self.grid_in
                    else:
                        inp_mark = [self.grid_out, self.grid_in]  

                if onestep_pred:
                    model_test_pred_len = 1
                else:
                    model_test_pred_len = self.args.loss_pred_len

                if flag in ['test', 'val'] and self.args.loss_pred_len < batch_y.shape[1]:
                    if onestep_pred:
                        all_predictions = np.full((batch_y.shape[0], batch_y.shape[1]+self.args.loss_pred_len,
                                                    self.args.loss_pred_len, batch_y.shape[2]), np.nan)
                        no_of_iters     = batch_y.shape[1]
                        for j in range(no_of_iters):
                            if j == 0:
                                inp    = batch_x.reshape(inp_shape)
                            else:
                                inp = torch.cat([inp[:, 1:, :], pred_forecast[:, :1, :]], dim=1).to(self.device)
                            pred = self.model(inp, inp_mark)
                            pred_forecast = pred

                            j_ = j%self.args.loss_pred_len
                            all_predictions[:, j:j+self.args.loss_pred_len, j_] = pred[:, 0:self.args.loss_pred_len].detach().cpu()
                            # mean_pred_forecast  = np.nanmean(all_predictions[:,j,:,:], axis=1, keepdims=True)
                            # pred_forecast       = torch.tensor(mean_pred_forecast).float().to(self.device)
                        pred    = np.nanmean(all_predictions, axis=2)
                        pred    = torch.tensor(pred[:,:batch_y.shape[1],:]).float()
                    else:
                        preds_temp = []
                        no_of_iters     = batch_y.shape[1]
                        for j in range(no_of_iters):
                            if j%self.args.loss_pred_len == 0:
                                if j == 0:
                                    inp    = batch_x.reshape(inp_shape)
                                else:
                                    inp = torch.cat([inp[:, self.args.loss_pred_len:, :],
                                                    pred_forecast[:, :self.args.loss_pred_len, :]], dim=1).to(self.device)
                            else:
                                continue
                            pred = self.model(inp, inp_mark)
                            pred_forecast = pred
                            preds_temp.append(pred)
                        pred    = torch.concatenate(preds_temp, dim=1)
                        pred    = pred[:,:batch_y.shape[1],:].detach().cpu()
                    truth_y = batch_y.detach().cpu()
                else:
                    inp    = batch_x.reshape(inp_shape)
                    pred = self.model(inp, inp_mark)
                    pred = pred.detach().cpu()
                    truth_y = batch_y.detach().cpu()

                    out_len_temp = min(self.args.loss_pred_len, batch_y.shape[1])
                    pred      = pred[:, 0:out_len_temp, :]
                    truth_y   = truth_y[:, 0:out_len_temp, :]

                all_preds.append(pred)
                all_trues.append(truth_y)

                if self.args.model == 'KRNO':
                    if self.args.data in ['m4', 'Cryptos', 'Traj'] and flag in ['test']:
                        loss = criterion(pred.to(self.device), batch_y.to(self.device), self.test_grid_out)
                    else:
                        loss = criterion(pred.to(self.device), batch_y.to(self.device), self.grid_out)
                    loss = loss.item()
                else:
                    loss = criterion(pred, truth_y)
                # print(i)

                eval_loss.append(loss)
        self.model.train()

        return np.sqrt(np.mean(eval_loss)), np.concatenate(all_preds, axis=0),\
                    np.concatenate(all_trues, axis=0)

    def train(self, setting, train_loader, test_loader, vali_loader=None, train_loss_optimal=None, plot_idx = 0,
                val_loss_min=np.Inf, train_loss_min =None, weight_decay_flag=False, train_without_val = False):

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        if weight_decay_flag:
            check_convergence = CheckConvergence(patience=self.args.patience, verbose=True,
                                                val_loss_min=val_loss_min,
                                                train_loss_optimal=train_loss_min)
        else:
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True,
                                            val_loss_min=val_loss_min)
        
        # if train_loss_optimal is not None:
        #     early_stopping = EarlyStopping(patience=self.args.patience, verbose=True,
        #                                     val_loss_min=train_loss_min)
        # else:
        #     check_convergence = CheckConvergence(patience=self.args.patience, verbose=True,
        #                                          val_loss_min=val_loss_min,
        #                                          train_loss_optimal=train_loss_min)

        criterion = self._select_criterion()
        step_size = 20
        gamma = 0.9
        scheduler = torch.optim.lr_scheduler.StepLR(self.model_optim, step_size=step_size, gamma=gamma)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        train_losses, val_losses, test_losses = [], [], []


        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_l2_full = 0

            if epoch==0:
                train_loss, _, _ = self.vali(train_loader, criterion, )
                if vali_loader is not None:
                    vali_loss, _, _ = self.vali(vali_loader, criterion, flag  = 'val')
                else:
                    vali_loss = None
                test_loss, _, _ = self.vali(test_loader, criterion, flag ='test')

                train_losses.append(train_loss)
                val_losses.append(vali_loss)
                test_losses.append(test_loss)

                if vali_loader is not None:
                    print("Epoch: {0}, Steps: {1} | Train rmse: {2:.7f} Vali rmse: {3:.7f} Test rmse: {4:.7f}".format(
                        epoch , train_steps, train_loss, vali_loss, test_loss))
                else:
                    print("Epoch: {0}, Steps: {1} | Train rmse: {2:.7f} Test rmse: {3:.7f}".format(
                        epoch, train_steps, train_loss, test_loss))
                if train_without_val:
                    early_stopping(train_loss, self.model, path, train_loss, epoch)

            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                loss = 0

                iter_count += 1
                self.model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # normailizing input times to [0, 1]
                batch_x_mark = batch_x_mark - batch_x_mark[:,0].reshape(-1,1)
                batch_x_mark = batch_x_mark/batch_x_mark[:, -1].reshape(-1,1) 

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Extract shape of the input tensor for reshaping (i.e. stacking the
                # time and channels dimension together)
                inp_shape = list(batch_x.shape)
                inp_shape = inp_shape[:-1]
                inp_shape.append(-1)
                
                batch_x_mark = batch_x_mark.reshape(inp_shape)
                # batch_y_mark = batch_y_mark.reshape(inp_shape)

                # Reshape input tensor into [b, input seq length , channels]
                inp    = batch_x.reshape(inp_shape)
                # inp_mark = torch.cat([batch_x_mark, batch_y_mark], dim=2).to(self.device)
                inp_mark = batch_x_mark

                # pdb.set_trace()
                # Model run
                if self.args.model == 'KRNO':
                    if self.args.seq_len == self.args.pred_len:
                        inp_mark = self.grid_in
                    else:
                        inp_mark = [self.grid_out, self.grid_in]   
                
                im = self.model(inp, inp_mark)

                ## 
                if hasattr(self.args, 'loss_pred_len'):
                    im      = im[:, 0:self.args.loss_pred_len, :]
                    batch_y = batch_y[:, 0:self.args.loss_pred_len, :]
                # Loss calculation
                if self.args.model == 'KRNO':
                    loss = criterion(im, batch_y, self.grid_out)
                else:
                    loss = criterion(im, batch_y)

                train_loss.append(loss.item())
                if (i + 1) % int(train_steps/5) == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.model_optim.step()

            # if self.args.model != 'KRNO':
            #     scheduler.step()
            # scheduler.step()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.sqrt(np.mean(train_loss))
            if vali_loader is not None:
                vali_loss, _, _ = self.vali(vali_loader, criterion, flag  = 'val')
            else:
                vali_loss = None
            test_loss, _, _ = self.vali(test_loader, criterion, flag  = 'test')

            train_losses.append(train_loss)
            val_losses.append(vali_loss)
            test_losses.append(test_loss)

            if vali_loader is not None:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, test_loss))
            
            if weight_decay_flag:
                check_convergence(train_loss, self.model, self.model_optim, path, vali_loss)
                if check_convergence.early_stop:
                    print("Train loss converged!!!")
                    break
            elif train_without_val:
                early_stopping(train_loss, self.model, path, train_loss, epoch)
                if early_stopping.early_stop:
                    print("Train loss converged!!!")
                    break
            elif train_loss_optimal is not None:
                early_stopping(train_loss, self.model, path, train_loss)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                convergence_flag = np.isclose(train_loss, train_loss_optimal,
                                       rtol=0.15, atol=1e-6,)
                if convergence_flag or train_loss < train_loss_optimal:
                    break
            else:
                early_stopping(vali_loss, self.model, path, train_loss, epoch)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            

            # if train_loss_optimal is not None:
            #     check_flag = np.isclose(train_loss_optimal, train_loss,
            #                            rtol=0.05)
            #     rel_change = np.abs(train_loss_optimal-train_loss)/np.abs(train_loss)
            #     print(f'Rel change in train loss: {rel_change}')
            #     if check_flag or train_loss < train_loss_optimal:
            #         print("Early stopping")
            #         break
            #     early_stopping(train_loss, self.model, path)
            # else:
            #     check_convergence(train_loss, self.model, self.model_optim, path, vali_loss)
            #     if check_convergence.early_stop:
            #         print("Train loss converged!!!")
            #         break

            # adjust_learning_rate(model_optim, epoch + 1, self.args)

        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))
        
        folder_path = f'./checkpoints/{setting}/'
        # plot train, val, test errors during training
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_losses, 'k*-', label='train', linewidth=1)
        if vali_loader is not None:
            ax.plot(val_losses, 'b^-', label='val', linewidth=1)
        ax.plot(test_losses, 'r+-', label='test', linewidth=1)
        ax.legend()
        ax.set_title('rmse values during training')
        plt.tight_layout()
        plt.grid()
        plt.savefig(os.path.join(folder_path, f'loss_plot_idx_{plot_idx}.pdf'), bbox_inches='tight')
        plt.yscale('log')
        plt.savefig(os.path.join(folder_path, f'loss_plot_log_idx_{plot_idx}.pdf'), bbox_inches='tight')

        # if train_loss_optimal is not None:
        #     return self.model, train_loss
        # else:
        #     return self.model, check_convergence.val_loss_min, check_convergence.train_loss_optimal
        if weight_decay_flag:
            val_loss_avg = np.mean(val_losses[-self.args.patience:])
            return self.model, val_loss_avg, check_convergence.train_loss_optimal
        elif train_without_val:
            return self.model
        else:
            return self.model, early_stopping.val_loss_min

