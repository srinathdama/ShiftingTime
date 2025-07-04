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
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_loader, criterion):
        total_loss = []
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
                
                model_test_pred_len = self.args.loss_pred_len

                if model_test_pred_len < batch_y.shape[1]:
                    # all_predictions shape is [b, test_output_length+loss_pred_len, loss_pred_len, channels]
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
                    truth_y = batch_y.detach().cpu()
                    
                else:
                    inp    = batch_x.reshape(inp_shape)
                    pred = self.model(inp, inp_mark)
                    pred = pred.detach().cpu()
                    truth_y = batch_y.detach().cpu()

                    out_len_temp = min(self.args.loss_pred_len, batch_y.shape[1])
                    pred      = pred[:, 0:out_len_temp, :]
                    truth_y   = truth_y[:, 0:out_len_temp, :]

                loss = criterion(pred, truth_y)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting, train_loader, vali_loader, test_loader, plot_idx=0):

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        train_losses, val_losses, test_losses = [], [], []


        for epoch in range(self.args.train_epochs):
            iter_count = 0

            if epoch==0:
                train_loss = self.vali(train_loader, criterion)
                vali_loss  = self.vali( vali_loader, criterion)
                test_loss  = self.vali( test_loader, criterion)

                train_losses.append(train_loss)
                val_losses.append(vali_loss)
                test_losses.append(test_loss)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
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
                im = self.model(inp, inp_mark)

                ## 
                if hasattr(self.args, 'loss_pred_len'):
                    im      = im[:, 0:self.args.loss_pred_len, :]
                    batch_y = batch_y[:, 0:self.args.loss_pred_len, :]
                # Loss calculation
                loss = criterion(im, batch_y)

                train_loss.append(loss.item())
                if (i + 1) % int(train_steps/5) == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.mean(train_loss)
            vali_loss = self.vali( vali_loader, criterion)
            test_loss = self.vali( test_loader, criterion)

            train_losses.append(train_loss)
            val_losses.append(vali_loss)
            test_losses.append(test_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

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


        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))

        return self.model, early_stopping.val_loss_min

    def test(self, test_data, test_loader, setting=None, onestep_pred=False, flag='test', test=0, plot_flag = False):
        if test:
            print('loading model')
            checkpoint = torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            self.model.load_state_dict(checkpoint['model_state_dict'])

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

        eval_loss = []
        all_preds = []
        all_trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
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
                if onestep_pred:
                    model_test_pred_len = 1
                else:
                    model_test_pred_len = self.args.loss_pred_len

                if model_test_pred_len < batch_y.shape[1]:
                    # all_predictions shape is [b, test_output_length+loss_pred_len, loss_pred_len, channels]
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
                    truth_y = batch_y.detach().cpu()
                    
                else:
                    inp    = batch_x.reshape(inp_shape)
                    pred = self.model(inp, inp_mark)
                    pred = pred.detach().cpu()
                    truth_y = batch_y.detach().cpu()

                    out_len_temp = min(self.args.loss_pred_len, batch_y.shape[1])
                    pred      = pred[:, 0:out_len_temp, :]
                    truth_y   = truth_y[:, 0:out_len_temp, :]

                if test_data.scale and self.args.inverse:
                    shape = pred.shape
                    pred = test_data.inverse_transform(pred.squeeze(0)).reshape(shape)
                    truth_y = test_data.inverse_transform(truth_y.squeeze(0)).reshape(shape)

                all_preds.append(pred)
                all_trues.append(truth_y)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], truth_y[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(all_preds)
        trues = np.array(all_trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        test_results = {'mse': mse, 'mae':mae}
        print('mse:{}, mae:{}'.format(mse, mae))
        folder_path = f'./checkpoints/{setting}'
        if test==0:
            model_type = 'final_model'
        else:
            model_type = 'best_model'

        f = open(os.path.join(folder_path, "result_long_term_forecast.txt"), 'a')
        f.write(setting + "  \n")
        f.write(model_type + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return test_results
