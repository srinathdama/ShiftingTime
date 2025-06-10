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



    def vali(self, vali_loader, criterion, setting=None, flag='train', plot_flag=False):
        total_loss = []
        if plot_flag:
            preds = []
            trues = []
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
                
                pred = self.model(inp, inp_mark)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                pred = pred[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = pred.detach().cpu()
                truth_y = batch_y.detach().cpu()
                if plot_flag:
                    preds.append(pred)
                    trues.append(truth_y)

                if hasattr(self.args, 'loss_pred_len'):
                    pred      = pred[:, 0:self.args.loss_pred_len, :]
                    truth_y   = truth_y[:, 0:self.args.loss_pred_len, :]

                if self.args.model == 'KRNO':
                    loss = criterion(pred.to(self.device), batch_y.to(self.device), self.grid_out)
                    loss = loss.item()
                else:
                    loss = criterion(pred, truth_y)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()

        if plot_flag:
            if flag=='test':
                folder_path = f'./checkpoints/{setting}/test_results/'
            elif flag=='train':
                folder_path = f'./checkpoints/{setting}/train_results/'
            elif flag=='val':
                folder_path = f'./checkpoints/{setting}/valid_results/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            ## plot the predictions
            fig, ax = visual_all(trues, preds, os.path.join(folder_path, f'{flag}_all_batches_without_forecasted.pdf'))
            return  total_loss, fig, ax
        else:
            return total_loss

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
        gamma = 0.75
        scheduler = torch.optim.lr_scheduler.StepLR(self.model_optim, step_size=step_size, gamma=gamma)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        train_losses, val_losses, test_losses = [], [], []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_l2_full = 0

            if epoch == 0:
                print("Epoch: {}".format(epoch))
                train_loss = self.vali(train_loader, criterion)
                if vali_loader is not None:
                    vali_loss = self.vali(vali_loader, criterion)
                else:
                    vali_loss = None
                test_loss = self.vali(test_loader, criterion)

                train_losses.append(train_loss)
                val_losses.append(vali_loss)
                test_losses.append(test_loss)

                if vali_loader is not None:
                    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                        epoch, train_steps, train_loss, vali_loss, test_loss))
                else:
                    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f}".format(
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
                if (i + 1) % 5 == 0:
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
            train_loss = np.average(train_loss)
            if vali_loader is not None:
                vali_loss = self.vali(vali_loader, criterion)
            else:
                vali_loss = None
            test_loss = self.vali(test_loader, criterion)

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
        train_loss, fig1, ax1 = self.vali(train_loader, criterion,setting=setting,
                               flag='train', plot_flag=True)
        if vali_loader is not None:
            vali_loss, fig2, ax2 = self.vali(vali_loader, criterion,setting=setting,
                               flag='val', plot_flag=True)
        
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
        #     return self.model, fig1, ax1, None, train_loss
        # else:
        #     return self.model, fig1, ax1, fig2, ax2, check_convergence.val_loss_min, check_convergence.train_loss_optimal

        if weight_decay_flag:
            val_loss_avg = np.mean(val_losses[-self.args.patience:])
            return self.model, val_loss_avg, check_convergence.train_loss_optimal
        elif train_without_val:
            return self.model
        elif train_loss_optimal is None:
            return self.model, fig1, ax1, fig2, ax2, \
                [early_stopping.val_loss_min, early_stopping.cur_train_loss_optimal, early_stopping.epoch]
        else:
            return self.model, fig1, ax1, None, None, [None, train_loss, None]

    def test(self, setting, test=0, flag='test', val_start_ratio=0):
        self.args.shuffle_flag = False
        self.args.temp_batch_size = 1
        test_data, test_loader = self._get_data(flag=flag)
        if test:
            print('loading model')
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            checkpoint = torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            self.model.load_state_dict(checkpoint['model_state_dict'])

            if flag=='val':
                val_start_idx = int(len(test_loader)*val_start_ratio)                

        # debugging
        # train_data, train_loader = self._get_data(flag='train')
        # criterion = self._select_criterion()
        # train_loss = self.vali(train_data, train_loader, criterion)

        preds = []
        trues = []
        save_freq   = 20
        series_total_len = test_data.data_x.shape[0]
        series_n_variates = test_data.data_x.shape[1]
        all_predictions = np.full((series_total_len-self.args.seq_len, self.args.pred_len, series_n_variates), np.nan)
        all_predictions_unnorm = np.full((series_total_len-self.args.seq_len, self.args.pred_len, series_n_variates), np.nan)
        if flag=='test':
            folder_path = f'./checkpoints/{setting}/test_results/'
        elif flag=='train':
            folder_path = f'./checkpoints/{setting}/train_results/'
        elif flag=='val':
            folder_path = f'./checkpoints/{setting}/valid_results/'
        
        if test==0:
            folder_path = folder_path + 'final_model'
        else:
            folder_path = folder_path + 'best_model'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if flag=='val' and test:
                    if i < val_start_idx:
                        continue
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # batch_y_mark = batch_y_mark - batch_x_mark[:,0].reshape(-1,1) 
                # batch_x_mark = batch_x_mark - batch_x_mark[:,0].reshape(-1,1) 

                # # normailizing output times to [0, 1] 
                # batch_y_mark = batch_y_mark - batch_x_mark[:,-1].reshape(-1,1) 
                # batch_y_mark = batch_y_mark/batch_y_mark[:, -1].reshape(-1,1) 

                # normailizing input times to [0, 1]
                batch_x_mark = batch_x_mark.float().numpy()
                batch_y_mark = batch_y_mark.float().numpy()

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

                ## use previous pred as input
                if i == 0 or flag=='train':
                # if i == 0 or flag in ['train', 'val']:
                    inp    = batch_x.reshape(inp_shape)
                else:
                    if flag == 'val' and test:
                        if i == val_start_idx:
                            inp    = batch_x.reshape(inp_shape)
                        else:
                            inp = torch.cat([inp[:, 1:, :], pred_forecast[:, :1, :]], dim=1).to(self.device)   
                    else:
                        inp = torch.cat([inp[:, 1:, :], pred_forecast[:, :1, :]], dim=1).to(self.device)

                # pdb.set_trace()
                # Model run
                if self.args.model == 'KRNO':
                    if self.args.seq_len == self.args.pred_len:
                        inp_mark = self.grid_in
                    else:
                        inp_mark = [self.grid_out, self.grid_in]      
                
                pred = self.model(inp, inp_mark)
                pred_forecast = pred

                # loss_ = loss_fn(pred, batch_y)
                # print(loss_.item())

                # f_dim = -1 if self.args.features == 'MS' else 0
                pred = pred[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                pred = pred.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if hasattr(self.args, 'loss_pred_len'):
                    pred            = pred[:, 0:self.args.loss_pred_len, :]
                    # pred_forecast   = pred_forecast[:, 0:self.args.loss_pred_len, :]
                if test_data.scale and self.args.inverse:
                    shape = pred.shape
                    if self.args.detrend_flag:
                        batch_y_mark = batch_y_mark[:, 0:self.args.loss_pred_len].reshape(-1,1)
                        pred = test_data.inverse_transform(pred.squeeze(0), batch_y_mark).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.squeeze(0), batch_y_mark).reshape(shape)
                    else:
                        pred = test_data.inverse_transform(pred.squeeze(0)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                pred = pred[:, :, :series_n_variates]
                batch_y = batch_y[:, :, :series_n_variates]

                truth = batch_y

                preds.append(pred)
                trues.append(truth)

                ##
                # if i==0:
                #     all_predictions[0:self.args.seq_len,0,:] = batch_x[0].detach().cpu().numpy()
                #     all_predictions[self.args.seq_len:self.args.seq_len+self.args.pred_len,0,:] = pred[0] 
                # else:
                if hasattr(self.args, 'loss_pred_len'):
                    j_ = i%self.args.loss_pred_len
                    all_predictions[i:i+self.args.loss_pred_len,j_,:] = pred[0] 
                    # all_predictions_unnorm[i:i+self.args.loss_pred_len,j_,:] = pred_forecast[0].detach().cpu()
                else:
                    j_ = i%self.args.pred_len
                    all_predictions[i:i+self.args.pred_len,j_,:] = pred[0] 
                    # all_predictions_unnorm[i:i+self.args.pred_len,j_,:] = pred_forecast[0].detach().cpu()
                # mean_pred_forecast  = np.nanmean(all_predictions_unnorm[i,:,:], axis=0, keepdims=True)
                # pred_forecast       = torch.tensor(mean_pred_forecast).float().to(self.device).unsqueeze(0)
                # if i % save_freq == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], truth[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print(f'{flag} shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print(f'{flag} shape:', preds.shape, trues.shape)

        predicted_forecast = np.concatenate((preds[:-1,0,0], preds[-1,:,0]))
        truth_forecast     = np.concatenate((trues[:-1,0,0], trues[-1,:,0]))
        test_results       = {'truth':truth_forecast, 'predicted_forecast':predicted_forecast,
                               'all_predictions': all_predictions}

        ## plot the predictions
        fig, ax = visual(truth_forecast, predicted_forecast,
                os.path.join(folder_path, f'{flag}_plot_forecasted_1step.pdf'),
                title=flag, multiple_preds=all_predictions)

        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        mean_all_prediction        = np.nanmean(all_predictions, axis=1).reshape(1,1,-1) # bs=1, steps = -1
        if flag == 'val' and test:
            mean_all_prediction = mean_all_prediction[:,:,val_start_idx:]
        llmtime_metrics            = Evaluator().evaluate(truth_forecast.reshape(1, -1), mean_all_prediction)
        test_results.update(llmtime_metrics)

        print('KRNO - mse:{}, mae:{}'.format(llmtime_metrics['mse'], llmtime_metrics['mae']))
        print('KRNO - nmse:{}, nmae:{}'.format(llmtime_metrics['nmse'], llmtime_metrics['nmae']))
        f = open(os.path.join(folder_path, "result_long_term_forecast.txt"), 'a')
        f.write(setting + "  \n")
        f.write("flag: " + flag + "  \n")
        f.write('Using mean of future predictions while sliding window by one step\n')
        f.write('\n')
        f.write('KRNO metrics \n')
        f.write('mse:{}, mae:{} \n'.format(llmtime_metrics['mse'],llmtime_metrics['mae']))
        f.write('nmse:{}, nmae:{} \n'.format(llmtime_metrics['nmse'],llmtime_metrics['nmae']))
        f.write('\n')
        f.write('\n')
        f.close()

        llmtime_metrics            = Evaluator().evaluate(truth_forecast.reshape(1, -1), predicted_forecast.reshape(1, 1,-1))
        print('KRNO - mse:{}, mae:{}'.format(llmtime_metrics['mse'], llmtime_metrics['mae']))
        print('KRNO - nmse:{}, nmae:{}'.format(llmtime_metrics['nmse'], llmtime_metrics['nmae']))
        f = open(os.path.join(folder_path, "result_long_term_forecast.txt"), 'a')
        f.write(setting + "  \n")
        f.write("flag: " + flag + "  \n")
        f.write('Using the first prediction while sliding window by one step\n')
        f.write('\n')
        f.write('KRNO metrics \n')
        f.write('mse:{}, mae:{} \n'.format(llmtime_metrics['mse'],llmtime_metrics['mae']))
        f.write('nmse:{}, nmae:{} \n'.format(llmtime_metrics['nmse'],llmtime_metrics['nmae']))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + f'{flag}_metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + f'{flag}_pred.npy', preds)
        np.save(folder_path + f'{flag}_true.npy', trues)

        return fig, ax, test_results
    
    def test_full_len(self, setting, test=0, flag='test', val_start_ratio=0):
        self.args.shuffle_flag = False
        self.args.temp_batch_size = 1
        test_data, test_loader = self._get_data(flag=flag)
        if test:
            print('loading model')
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            checkpoint = torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            self.model.load_state_dict(checkpoint['model_state_dict'])

            if flag=='val':
                val_start_idx = int(len(test_loader)*val_start_ratio)                

        # debugging
        # train_data, train_loader = self._get_data(flag='train')
        # criterion = self._select_criterion()
        # train_loss = self.vali(train_data, train_loader, criterion)

        preds = []
        trues = []
        save_freq   = 20
        series_n_variates = test_data.data_x.shape[1]
        if flag=='test':
            folder_path = f'./checkpoints/{setting}/test_results/'
        elif flag=='train':
            folder_path = f'./checkpoints/{setting}/train_results/'
        elif flag=='val':
            folder_path = f'./checkpoints/{setting}/valid_results/'
        
        if test==0:
            folder_path = folder_path + 'final_model'
        else:
            folder_path = folder_path + 'best_model'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if flag=='val' and test:
                    if i < val_start_idx:
                        continue
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # normailizing input times to [0, 1]
                batch_x_mark = batch_x_mark.float().numpy()
                batch_y_mark = batch_y_mark.float().numpy()

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

                ## use previous pred as input
                if i%self.args.loss_pred_len == 0:
                    if i == 0:
                        inp    = batch_x.reshape(inp_shape)
                    else:
                        inp = torch.cat([inp[:, self.args.loss_pred_len:, :],
                                        pred_forecast[:, :self.args.loss_pred_len, :]], dim=1).to(self.device)
                else:
                    continue

                # pdb.set_trace()
                # Model run
                if self.args.model == 'KRNO':
                    if self.args.seq_len == self.args.pred_len:
                        inp_mark = self.grid_in
                    else:
                        inp_mark = [self.grid_out, self.grid_in]      
                
                pred = self.model(inp, inp_mark)
                pred_forecast = pred

                # loss_ = loss_fn(pred, batch_y)
                # print(loss_.item())

                # f_dim = -1 if self.args.features == 'MS' else 0
                pred = pred[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                pred = pred.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if hasattr(self.args, 'loss_pred_len'):
                    pred            = pred[:, 0:self.args.loss_pred_len, :]
                    # pred_forecast   = pred_forecast[:, 0:self.args.loss_pred_len, :]
                if test_data.scale and self.args.inverse:
                    shape = pred.shape
                    if self.args.detrend_flag:
                        batch_y_mark = batch_y_mark[:, 0:self.args.loss_pred_len].reshape(-1,1)
                        pred = test_data.inverse_transform(pred.squeeze(0), batch_y_mark).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.squeeze(0), batch_y_mark).reshape(shape)
                    else:
                        pred = test_data.inverse_transform(pred.squeeze(0)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                pred = pred[:, :, :series_n_variates]
                batch_y = batch_y[:, :, :series_n_variates]

                truth = batch_y

                preds.append(pred)
                trues.append(truth)

        preds = np.concatenate(preds, axis=1)
        trues = np.concatenate(trues, axis=1)
        print(f'{flag} shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print(f'{flag} shape:', preds.shape, trues.shape)

        predicted_forecast = preds.reshape(-1)
        truth_forecast     = trues.reshape(-1)
        test_results       = {'truth':truth_forecast, 'predicted_forecast':predicted_forecast}

        ## plot the predictions
        fig, ax = visual(truth_forecast, predicted_forecast,
                os.path.join(folder_path, f'{flag}_plot_forecasted_full_window.pdf'),
                title=flag)

        llmtime_metrics            = Evaluator().evaluate(truth_forecast.reshape(1, -1), predicted_forecast.reshape(1, 1,-1))
        test_results.update(llmtime_metrics)

        print('KRNO - mse:{}, mae:{}'.format(llmtime_metrics['mse'], llmtime_metrics['mae']))
        print('KRNO - nmse:{}, nmae:{}'.format(llmtime_metrics['nmse'], llmtime_metrics['nmae']))
        f = open(os.path.join(folder_path, "result_long_term_forecast.txt"), 'a')
        f.write(setting + "  \n")
        f.write("flag: " + flag + "  \n")
        f.write('Using the full predictions while sliding window by loss_pred_length \n')
        f.write('\n')
        f.write('KRNO metrics \n')
        f.write('mse:{}, mae:{} \n'.format(llmtime_metrics['mse'],llmtime_metrics['mae']))
        f.write('nmse:{}, nmae:{} \n'.format(llmtime_metrics['nmse'],llmtime_metrics['nmae']))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + f'{flag}_pred.npy', preds)
        np.save(folder_path + f'{flag}_true.npy', trues)

        return fig, ax, test_results
    



