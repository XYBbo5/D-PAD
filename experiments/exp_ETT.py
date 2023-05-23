import os
import time

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import warnings

from model.D_PAD_GAT import DPAD_GAT
from model.D_PAD_ATT import DPAD_ATT
from model.D_PAD_SEBlock import DPAD_SE
from model.D_PAD_adpGCN import DPAD_GCN


from utils.ETTh_metrics import metric, metric_
from utils.tools import EarlyStopping, adjust_learning_rate, load_model, save_model, visual
warnings.filterwarnings('ignore')
from data_process.ETT_data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute
from experiments.exp_basic import Exp_Basic


class Exp_ETT(Exp_Basic):
    def __init__(self, args):
        super(Exp_ETT, self).__init__(args)
        self.test_loader = self._get_data(flag = 'test')
        

    def _build_model(self):
        if self.args.features == 'S':
            self.input_dim = 1
        elif self.args.features == 'M':
            if "ETT" in self.args.data:
                self.input_dim = 7
            elif self.args.data == 'ECL' or self.args.data == 'electricity':
                self.input_dim = 321
            elif self.args.data == 'solar_AL':
                self.input_dim = 137
            elif self.args.data == 'exchange':
                self.input_dim = 8
            elif self.args.data == 'traffic':
                self.input_dim = 862
            elif self.args.data == 'weather':
                self.input_dim = 21
            elif self.args.data == 'illness':
                self.input_dim = 7
        else:
            print('Error!')

        if 'DPAD_GAT' in self.args.model_name:
            model = DPAD_GAT(
                output_len=self.args.pred_len,
                input_len=self.args.seq_len,
                input_dim=self.input_dim,
                enc_hidden = self.args.enc_hidden,
                dec_hidden = self.args.dec_hidden,
                num_levels=self.args.levels,
                dropout=self.args.dropout,
                K_IMP=self.args.K_IMP,
                RIN=self.args.RIN
            )
        elif 'DPAD_ATT' in self.args.model_name:
            model = DPAD_ATT(
                output_len=self.args.pred_len,
                input_len=self.args.seq_len,
                input_dim=self.input_dim,
                enc_hidden = self.args.enc_hidden,
                dec_hidden = self.args.dec_hidden,
                num_levels=self.args.levels,
                dropout=self.args.dropout,
                K_IMP=self.args.K_IMP,
                RIN=self.args.RIN,
                num_heads = self.args.num_heads
            )
        elif 'DPAD_SE' in self.args.model_name:
            model = DPAD_SE(
                output_len=self.args.pred_len,
                input_len=self.args.seq_len,
                input_dim=self.input_dim,
                enc_hidden = self.args.enc_hidden,
                dec_hidden = self.args.dec_hidden,
                num_levels=self.args.levels,
                dropout=self.args.dropout,
                K_IMP=self.args.K_IMP,
                RIN=self.args.RIN
            )
        elif 'DPAD_GCN' in self.args.model_name:
            model = DPAD_GCN(
                output_len=self.args.pred_len,
                input_len=self.args.seq_len,
                input_dim=self.input_dim,
                enc_hidden = self.args.enc_hidden,
                dec_hidden = self.args.dec_hidden,
                num_levels=self.args.levels,
                dropout=self.args.dropout,
                K_IMP=self.args.K_IMP,
                RIN=self.args.RIN
            )

        # print(model)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'weather':Dataset_Custom,
            'ECL':Dataset_Custom,
            'electricity':Dataset_Custom,
            'Solar':Dataset_Custom,
            'traffic':Dataset_Custom,
            'exchange':Dataset_Custom,
            'illness':Dataset_Custom
        }
        Data = data_dict[self.args.data]

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size 
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag, 
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
            )

        return data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return model_optim

    def _select_criterion(self, losstype):
        if losstype == "mse":
            criterion = nn.MSELoss()
        elif losstype == "mae":
            criterion = nn.L1Loss()
        else:
            criterion = nn.L1Loss()
        return criterion

    def train(self, setting):
        train_loader = self._get_data(flag = 'train')
        valid_loader = self._get_data(flag = 'val')
        self.test_loader = self._get_data(flag = 'test')
        path = os.path.join(self.args.checkpoints, setting)
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=0.0002)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion(self.args.loss)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        epoch_start = 0

        for epoch in range(epoch_start, self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch_DPAD(batch_x, batch_y)

                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    print('use amp')
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            print('--------start to validate-----------')
            valid_loss = self.valid(valid_loader, criterion, flag="valid")
            print('--------start to test-----------')
            test_loss = self.valid(self.test_loader, criterion, flag="test")

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} valid Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, valid_loss, test_loss))

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Test Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, test_loss))



            early_stopping(valid_loss, self.model, path)
            # early_stopping(test_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            lr = adjust_learning_rate(model_optim, epoch+1, self.args)

        save_model(epoch, lr, self.model, path, model_name=self.args.data, horizon=self.args.pred_len)
        best_model_path = path+'/'+'checkpoint.pth'
        
        return best_model_path

    def valid(self, valid_loader, criterion, flag):
        self.model.eval()
        total_loss = []

        mses = []
        maes = []


        for i, (batch_x, batch_y) in enumerate(valid_loader):
            pred, true = self._process_one_batch_DPAD(batch_x, batch_y)

            mae, mse = metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())
            mses.append(mse)
            maes.append(mae)

            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)


        total_loss = np.average(total_loss)
        mse = np.average(mses)
        mae = np.average(maes)


        print('-----------start to {} {}-----------\n|  Normed  | mse:{:5.4f} | mae:{:5.4f} |'.format(self.args.rank, flag, mse, mae))
        
        return total_loss


    def test(self, setting, evaluate=0):
        
        self.model.eval()
        
        preds = []
        trues = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


        if evaluate:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        for i, (batch_x, batch_y) in enumerate(self.test_loader):
            pred, true = self._process_one_batch_DPAD(batch_x, batch_y)
            pred = pred.detach().cpu().numpy()
            true = true.detach().cpu().numpy()

            preds.append(pred)
            trues.append(true)


        preds = np.array(preds)
        trues = np.array(trues)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])


        mae, mse, rmse, mape, mspe, corr = metric_(preds, trues)
        print('|  Normed  | mse:{:5.4f} | mae:{:5.4f} | rmse:{:5.4f} | mape:{:5.4f} | mspe:{:5.4f} | corr:{:5.4f} |'.format(mse, mae, rmse, mape, mspe, corr))
                
        # result save
        if self.args.save:
            folder_path = 'exp/ETT_results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            mae, mse, rmse, mape, mspe, corr = metric_(preds, trues)
            print('Test:mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
      

        return mse, mae 


    def _process_one_batch_DPAD(self,batch_x, batch_y):
        batch_x = batch_x.float().to(self.args.rank)
        batch_y = batch_y.float()

        outputs = self.model(batch_x)

        f_dim = -1 if self.args.features=='MS' else 0
        # batch_y
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.args.rank)

        return outputs, batch_y


        
