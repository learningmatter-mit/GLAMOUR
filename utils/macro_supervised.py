import pandas as pd
import os
import errno
import json
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import shutil
import tempfile
import dgl
import torch
from utils.macro_dataset import MacroDataset
from utils.stopper import Stopper_v2
from utils.meter import Meter_v2
from torch.utils.data import DataLoader

import seaborn as sns
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap



from sklearn.metrics import auc

from dgllife.utils import RandomSplitter

from torch.optim import Adam



class MacroSupervised():
    def __init__(self, MacroDataset, MON_SMILES, BOND_SMILES, FEAT, FP_BITS_MON, FP_BITS_BOND, SEED, MODEL, SPLIT, NUM_EPOCHS, NUM_WORKERS, CUSTOM_PARAMS, MODEL_PATH=None, SAVE_MODEL=False, SAVE_OPT=False, SAVE_CONFIG=False):
        '''
        Initializes a MacroSupervised object
        
        Args:
        MacroDataset: MacroDataset, MacroDataset object for DGL Dataset
        MON_SMILES: str, path to .txt file of all monomers that comprise macromolecule and corresponding SMILES
        BOND_SMILES: str, path to .txt file of all bonds that comprise macromolecule and corresponding SMILES
        FEAT: str, type of attribute with which to featurizer nodes and edges of macromolecule
        FP_BITS_MON: int, size of fingerprint bit-vector for monomer 
        FP_BITS_BOND: int, size of fingerprint bit-vector for bond
        SEED: int, random seed for shuffling dataset
        MODEL: str, model architecture for supervised learning 
        SPLIT: str, proportion of the dataset to use for training, validation and test
        NUM_EPOCHS: int, maximum number of epochs allowed for training
        NUM_WORKERS: int, number of processes for data loading
        CUSTOM_PARAMS: dict, dictionary of custom hyperparameters
        MODEL_PATH: str, path to save models and configuration files (default=None)
        SAVE_MODEL: boolean, whether to save full model file (default=False)
        SAVE_OPT: boolean, whether to save optimizer files (default=False)
        SAVE_CONFIG: boolean, whether to save configuration file (default=False)
        
        Attributes:
        train_set: Subset, Subset of graphs for model training
        val_set: Subset, Subset of graphs for model validation
        test_set: Subset, Subset of graphs used for model testing
        model_load: dgllife model, Predictor with set hyperparameters
        
        '''
        np.random.seed(0)
        
        self._dataset = MacroDataset
        self._mon_smiles = pd.read_csv(MON_SMILES)
        self._bond_smiles = pd.read_csv(BOND_SMILES)
        self._feat = FEAT
        self._fp_bits_mon = FP_BITS_MON
        self._fp_bits_bond = FP_BITS_BOND
        self._seed = SEED
        self._model_name = MODEL
        self._split = SPLIT
        self._num_epochs = NUM_EPOCHS
        self._num_workers = NUM_WORKERS
        self._custom_params = CUSTOM_PARAMS
        self._model_path = MODEL_PATH
        self._save_model = SAVE_MODEL
        self._save_opt = SAVE_OPT
        self._save_config = SAVE_CONFIG
        self._normalizer = self._dataset.normalizer
        
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        
        if self._dataset.task == 'classification':
            if self._dataset.classtype == 'binary':
                self._loss_criterion = nn.BCEWithLogitsLoss(reduction='none')
            elif self._dataset.classtype == 'multiclass':
                self._loss_criterion = nn.CrossEntropyLoss()
            elif self._dataset.classtype == 'multilabel':
                self._loss_criterion = nn.BCEWithLogitsLoss(reduction='none')
        elif self._dataset.task == 'regression':
            self._loss_criterion = nn.SmoothL1Loss(reduction='none')
            
        self._config_update()
        if self._model_path != None:
            self._mkdir_p()
        self._split_dataset()
        self._load_hparams()
    
    def _config_update(self):
        ''' Utility function for update of configuration dictionary '''
        self._exp_config = {}
        self._exp_config['model'] = self._model_name
        self._exp_config['n_tasks'] = self._dataset.n_tasks
        self._exp_config['featurizer_type'] = self._feat
        if self._feat == 'fp':
            self._exp_config['in_node_feats'] = self._fp_bits_mon
            self._exp_config['in_edge_feats'] = self._fp_bits_bond
        elif self._feat == 'onehot':
            self._exp_config['in_node_feats'] = len(self._mon_smiles['Molecule'].tolist())
            self._exp_config['in_edge_feats'] = len(self._bond_smiles['Molecule'].tolist())
    
    def _mkdir_p(self):
        ''' Utility function for creation of folder for given path'''
        try:
            os.makedirs(self._model_path)
            print('Created directory {}'.format(self._model_path))
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(self._model_path):
                print('Directory {} already exists.'.format(self._model_path))
            else:
                raise
                
    def _split_dataset(self):
        ''' Utility function for splitting Dataset into Subsets for train, validation, and testing '''
        train_ratio, val_ratio, test_ratio = map(float, self._split.split(','))
        self.train_set, self.val_set, self.test_set = RandomSplitter.train_val_test_split(
            self._dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio, random_state=self._seed)
        
    def _load_hparams(self):
        ''' Utility function for loading default hyperparameters and updating them to reflect custom hyperparameters '''
        with open('./model_hparams/{}.json'.format(self._model_name), 'r') as f:
            config = json.load(f)
        config.update(self._custom_params)
        self._exp_config.update(config)
        
    def _load_model(self):
        ''' Utility function for loading model 
        
        Returns:
        model: dgllife model, Predictor with set hyperparameters
        '''
        if self._model_name == 'GCN':
            from dgllife.model import GCNPredictor
            model = GCNPredictor(
                in_feats=self._exp_config['in_node_feats'],
                hidden_feats=[self._exp_config['gnn_hidden_feats']] * self._exp_config['num_gnn_layers'],
                activation=[F.relu] * self._exp_config['num_gnn_layers'],
                residual=[self._exp_config['residual']] * self._exp_config['num_gnn_layers'],
                batchnorm=[self._exp_config['batchnorm']] * self._exp_config['num_gnn_layers'],
                dropout=[self._exp_config['dropout']] * self._exp_config['num_gnn_layers'],
                predictor_hidden_feats=self._exp_config['predictor_hidden_feats'],
                predictor_dropout=self._exp_config['dropout'],
                n_tasks=self._exp_config['n_tasks'])
        elif self._model_name == 'GAT':
            from dgllife.model import GATPredictor
            model = GATPredictor(
                in_feats=self._exp_config['in_node_feats'],
                hidden_feats=[self._exp_config['gnn_hidden_feats']] * self._exp_config['num_gnn_layers'],
                num_heads=[self._exp_config['num_heads']] * self._exp_config['num_gnn_layers'],
                feat_drops=[self._exp_config['dropout']] * self._exp_config['num_gnn_layers'],
                attn_drops=[self._exp_config['dropout']] * self._exp_config['num_gnn_layers'],
                alphas=[self._exp_config['alpha']] * self._exp_config['num_gnn_layers'],
                residuals=[self._exp_config['residual']] * self._exp_config['num_gnn_layers'],
                predictor_hidden_feats=self._exp_config['predictor_hidden_feats'],
                predictor_dropout=self._exp_config['dropout'],
                n_tasks=self._exp_config['n_tasks']
            )
        elif self._model_name == 'Weave':
            from dgllife.model import WeavePredictor
            model = WeavePredictor(
                node_in_feats=self._exp_config['in_node_feats'],
                edge_in_feats=self._exp_config['in_edge_feats'],
                num_gnn_layers=self._exp_config['num_gnn_layers'],
                gnn_hidden_feats=self._exp_config['gnn_hidden_feats'],
                graph_feats=self._exp_config['graph_feats'],
                gaussian_expand=self._exp_config['gaussian_expand'],
                n_tasks=self._exp_config['n_tasks']
            )
        elif self._model_name == 'MPNN':
            from dgllife.model import MPNNPredictor
            model = MPNNPredictor(
                node_in_feats=self._exp_config['in_node_feats'],
                edge_in_feats=self._exp_config['in_edge_feats'],
                node_out_feats=self._exp_config['node_out_feats'],
                edge_hidden_feats=self._exp_config['edge_hidden_feats'],
                num_step_message_passing=self._exp_config['num_step_message_passing'],
                num_step_set2set=self._exp_config['num_step_set2set'],
                num_layer_set2set=self._exp_config['num_layer_set2set'],
                n_tasks=self._exp_config['n_tasks']
            )
        elif self._model_name == 'AttentiveFP':
            from dgllife.model import AttentiveFPPredictor
            model = AttentiveFPPredictor(
                node_feat_size=self._exp_config['in_node_feats'],
                edge_feat_size=self._exp_config['in_edge_feats'],
                num_layers=self._exp_config['num_layers'],
                num_timesteps=self._exp_config['num_timesteps'],
                graph_feat_size=self._exp_config['graph_feat_size'],
                dropout=self._exp_config['dropout'],
                n_tasks=self._exp_config['n_tasks']
            )
        return model
    
    def _collate_molgraphs(self, data):
        ''' Utility function for batching list of datapoints for Dataloader 
        
        Args:
        data : list, list of 4-tuples, each for a single datapoint consisting of an ID, a DGLGraph, all-task labels and a binary
        mask indicating the existence of labels
        
        Returns:
        IDs : list, list of GBIDs
        bg : DGLGraph, batched DGLGraph.
        labels : Tensor of dtype float32 and shape (len(data), data.n_tasks), batched datapoint labels
        masks : Tensor of dtype float32 and shape (len(data), data.n_tasks), batched datapoint binary mask indicating the
        existence of labels.
        '''
        IDs, graphs, labels, masks = map(list, zip(*data))

        bg = dgl.batch(graphs)
        bg.set_n_initializer(dgl.init.zero_initializer)
        bg.set_e_initializer(dgl.init.zero_initializer)
        labels = torch.stack(labels, dim=0)

        if masks is None:
            masks = torch.ones(labels.shape)
        else:
            masks = torch.stack(masks, dim=0)

        return IDs, bg, labels, masks

    def _run_a_train_epoch(self, epoch, model, data_loader, optimizer):
        ''' Utility function for running a train epoch 
        
        Args:
        epoch : int, training epoch count
        model : dgllife model, Predictor with set hyperparameters
        data_loader : DataLoader, DataLoader for train, validation, or test
        optimizer : torch.optim.Adam, Adam object
        '''
        model.train()
        train_meter = Meter_v2()
        for batch_id, batch_data in enumerate(data_loader):
            IDs, bg, labels, masks = batch_data

            labels, masks = labels.to(self._device), masks.to(self._device)
            logits = self._predict(model, bg)
            if self._dataset.task == 'classification':
                if self._dataset.classtype == 'multiclass':
                    losslabels = torch.max(labels, 1)[1]
                else:
                    losslabels = labels
            else:
                losslabels = labels
            loss = (self._loss_criterion(logits, losslabels) * (masks != 0).float()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_meter.update(logits, labels, masks)
            if batch_id % 20 == 0:
                if self._dataset.task == 'classification':
                    print_val = loss.item()
                elif self._dataset.task == 'regression':
                    print_val = np.mean(train_meter.compute_metric('rmse'))
                print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
                    epoch + 1, self._num_epochs, batch_id + 1, len(data_loader), print_val))
        
    def _run_an_eval_epoch(self, model, data_loader):
        ''' Utility function for running an evaluation (validation/test) epoch
        
        Args:
        model : dgllife model, Predictor with set hyperparameters
        data_loader : DataLoader, DataLoader for train, validation, or test
        
        Returns:
        metric_dict : dict, dictionary of metric names and corresponding evaluation values
        '''
        model.eval()
        eval_meter = Meter_v2()
        loss_list = []
        with torch.no_grad():
            for batch_id, batch_data in enumerate(data_loader):
                IDs, bg, labels, masks = batch_data
                labels, masks = labels.to(self._device), masks.to(self._device)
                logits = self._predict(model, bg)
                eval_meter.update(logits, labels, masks)
                if self._dataset.task == 'classification':
                    if self._dataset.classtype == 'multiclass':
                        losslabels = torch.max(labels, 1)[1]
                    else:
                        losslabels = labels
                else:
                    losslabels = labels
                loss = (self._loss_criterion(logits, losslabels) * (masks != 0).float()).mean()
                loss_list.append(loss.item())
        if self._dataset.task == 'classification':
            plotvals = eval_meter.compute_metric('roc_curve')
            if self._dataset.classtype == 'binary':
                metric_dict = {'loss': np.mean(loss_list), 'ROC-AUC': np.mean(eval_meter.compute_metric('roc_auc_score')), 'F1': np.mean(eval_meter.compute_metric('f1_score')), 'recall': np.mean(eval_meter.compute_metric('recall_score')), 'precision': np.mean(eval_meter.compute_metric('precision_score')), 'accuracy': np.mean(eval_meter.compute_metric('accuracy_score'))}
            elif self._dataset.classtype == 'multiclass':
                metric_dict = {'loss': np.mean(loss_list), 'ROC-AUC': np.mean(eval_meter.compute_metric('roc_auc_score')), 'F1': np.mean(eval_meter.compute_metric('f1_score')), 'recall': np.mean(eval_meter.compute_metric('recall_score')), 'precision': np.mean(eval_meter.compute_metric('precision_score')), 'accuracy': np.mean(eval_meter.compute_metric('accuracy_score'))}
            elif self._dataset.classtype == 'multilabel':
                metric_dict = {'loss': np.mean(loss_list), 'ROC-AUC': np.mean(eval_meter.compute_metric('roc_auc_score')), 'F1': np.mean(eval_meter.compute_metric('f1_score')), 'recall': np.mean(eval_meter.compute_metric('recall_score')), 'precision': np.mean(eval_meter.compute_metric('precision_score')), 'accuracy': np.mean(eval_meter.compute_metric('accuracy_score')), 'hamming loss': np.mean(eval_meter.compute_metric('hamming_loss'))}
        elif self._dataset.task == 'regression':
            plotvals = eval_meter.inverse(self._normalizer)
            metric_dict = {'rmse': np.mean(eval_meter.compute_metric('rmse')), 'L1loss': np.mean(loss_list), 'r2': np.mean(eval_meter.compute_metric('r2')), 'mae': np.mean(eval_meter.compute_metric('mae')), 'spearmanr': np.mean(eval_meter.compute_metric('spearmanr')), 'kendalltau': np.mean(eval_meter.compute_metric('kendalltau'))}
    
        return metric_dict, plotvals
        
    def _predict(self, model, bg):
        ''' Utility function for moving batched graph and node/edge feats to device
        
        Args:
        model : dgllife model, Predictor with set hyperparameters
        bg : DGLGraph, batched DGLGraph
        
        Returns:
        model(bg, node_feats, edge_feats) : model moved to device
        '''
        bg = bg.to(self._device)
        if self._model_name in ['GCN', 'GAT']:
            node_feats = bg.ndata.pop('h').to(self._device)
            return model(bg, node_feats)
        else:
            node_feats = bg.ndata.pop('h').to(self._device)
            edge_feats = bg.edata.pop('e').to(self._device)
            return model(bg, node_feats, edge_feats)
    
    def main(self):
        ''' Performs training, validation, and testing of dataset with output of metrics to centralized files'''
        train_loader = DataLoader(dataset=self.train_set, batch_size=self._exp_config['batch_size'], shuffle=True,
                              collate_fn=self._collate_molgraphs, num_workers=self._num_workers)
        val_loader = DataLoader(dataset=self.val_set, batch_size=self._exp_config['batch_size'],
                            collate_fn=self._collate_molgraphs, num_workers=self._num_workers)
        test_loader = DataLoader(dataset=self.test_set, batch_size=self._exp_config['batch_size'],
                             collate_fn=self._collate_molgraphs, num_workers=self._num_workers)
        
        self.model_load = self._load_model()
        model = self.model_load.to(self._device)
        
        if self._model_path == None:
            tmp_dir = tempfile.mkdtemp()
            tmppath = tempfile.NamedTemporaryFile(prefix='model',suffix='.pth',dir=tmp_dir)
        else:
            tmppath = tempfile.NamedTemporaryFile(prefix='model',suffix='.pth',dir=self._model_path)
        
        optimizer = Adam(model.parameters(), lr=self._exp_config['lr'],
                     weight_decay=self._exp_config['weight_decay'])
        stopper = Stopper_v2(savepath=self._model_path, mode='lower', patience=self._exp_config['patience'], 
                            filename=tmppath.name)

        for epoch in range(self._num_epochs):
            self._run_a_train_epoch(epoch, model, train_loader, optimizer)

            val_score = self._run_an_eval_epoch(model, val_loader)[0]
            early_stop = stopper.step(
                val_score[list(val_score.keys())[0]], 
                model, optimizer, self._model_name, self._save_model, self._save_opt)
            
            print('epoch {:d}/{:d}, validation {} {:.4f}, best validation {} {:.4f}, '.format(
                epoch + 1, self._num_epochs, list(val_score.keys())[0],
                val_score[list(val_score.keys())[0]], 
                list(val_score.keys())[0], stopper.best_score) +
                  ', '.join('validation {} {:.4f}'.format(
                      list(val_score.keys())[num], val_score[list(val_score.keys())[num]]) for num in 
                            range(1,len(list(val_score.keys())))))

            if early_stop:
                break

        stopper.load_checkpoint(model)
        tmppath.close()
        if self._model_path == None:
            shutil.rmtree(tmp_dir)
        self._best_val_score = self._run_an_eval_epoch(model, val_loader)[0]
        self._val_plotvals = self._run_an_eval_epoch(model, val_loader)[1]
        self._test_score = self._run_an_eval_epoch(model, test_loader)[0]
        self._test_plotvals = self._run_an_eval_epoch(model, test_loader)[1]
        
        print(', '.join('best val {} {:.4f}'.format(
            list(self._best_val_score.keys())[num], self._best_val_score[list(self._best_val_score.keys())[num]]) for num in
                        range(0,len(list(self._best_val_score.keys())))))
        print(', '.join('best val {} {:.4f}'.format(
            list(self._test_score.keys())[num], self._test_score[list(self._test_score.keys())[num]]) for num in 
                        range(0,len(list(self._test_score.keys())))))
    
        if self._save_config == True:
            with open(self._model_path + '/configure.json', 'w') as f:
                json.dump(self._exp_config, f, indent=2)
        
        print('\nBest Validation Metrics:')
        print('\n'.join('{} {:.4f}'.format(
            list(self._best_val_score.keys())[num], self._best_val_score[list(self._best_val_score.keys())[num]]) for num in
                        range(1,len(list(self._best_val_score.keys())))))
        
        print('\nTest Metrics:')
        print('\n'.join('{} {:.4f}'.format(
            list(self._test_score.keys())[num], self._test_score[list(self._test_score.keys())[num]]) for num in 
                        range(1,len(list(self._test_score.keys())))))
        
        self.model = model
        
    def rocauc_plot(self, plottype, fig_path):
        ''' Plots ROC-AUC curve for classification task
        
        Args:
        plottype : str, dataset to plot, 'val' for validation or 'test' for test
        fig_path : str, path to save figure
        '''
        mean_fpr = self._val_plotvals[0]
        mean_tpr = self._val_plotvals[1]
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)

        plt.figure()
        lw = 2
        plt.plot(mean_fpr, mean_tpr, color='#2C7FFF',lw=lw)
        plt.plot([0, 1], [0, 1], color='#B2B2B2', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=18)
        plt.ylabel('True Positive Rate', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.text(0.95, 0.03, 'ROC-AUC = %0.2f' % (mean_auc),
        verticalalignment='bottom', horizontalalignment='right',
        fontsize=18)
        
        plt.tight_layout()
        plt.show()
        plt.savefig(fig_path + 'ROC_AUC.pdf')
        
    def parity_plot(self, plottype, fig_path):
        ''' Makes parity plot for regression task
        
        Args:
        plottype : str, dataset to plot, 'val' for validation or 'test' for test
        fig_path : str, path to save figure
        '''
        plt.plot([0.01, 10000], [0.01, 10000], 'k--', lw=1)
        
        if plottype == 'val':
            y_true = self._val_plotvals[0]
            y_pred = self._val_plotvals[1]
            r2_num = self._best_val_score['r2']
        elif plottype == 'test':
            y_true = self._test_plotvals[0]
            y_pred = self._test_plotvals[1]
            r2_num = self._test_score['r2']
        
        removed = []
        for i in range(len(y_true)):
            if float(y_pred[i]) <= 0:
                removed.append(i)
        for val in reversed(removed):
            y_true.pop(val)
            y_pred.pop(val)
        
        colors = ["#2C7FFF", "#B2B2B2", "#B12122"]
        cmap = LinearSegmentedColormap.from_list("mycmap", colors)
        plt.hexbin(
            y_true, y_pred, mincnt=1, cmap=cmap, vmin=2, vmax=5, xscale='log', yscale='log', bins='log')
    
        plt.xlabel('Ground Truth Activity', fontsize=18)
        plt.ylabel('Predicted Activity', fontsize=18)
        plt.xlim([0.01, 10000])
        plt.ylim([0.01, 10000])
        plt.tick_params(axis='both', which='major', labelsize=16)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label(label='Density', size=18)
        plt.tight_layout()
        plt.text(6, 0.015, 'R$^2$ = %0.3f' % (r2_num),
            verticalalignment='bottom', horizontalalignment='right',
            fontsize=18)
        plt.show()
        plt.savefig(fig_path + 'parityplot.pdf')
        plt.close()
        
