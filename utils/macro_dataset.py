import pandas as pd
import numpy as np

import dgl
import torch

from sklearn.preprocessing import StandardScaler, QuantileTransformer


class MacroDataset():
    def __init__(self, DF_PATH, SEED, TASK, LABELNAME, MODEL, NX_GRAPHS, NORM=None):
        '''
        Initializes a MacroDataset object
        
        Args:
        DF_PATH: str, path to DataFrame containing all macromolecules and corresponding labels
        SEED: int, random seed for shuffling dataset
        LABELNAME: str, name of label to classify
        NX_GRAPHS: dict, dictionary of featurized NetworkX graph for each macromolecule ID
        NORM: str, normalization method for regression dataset (default=None)
        
        Attributes:
        IDs: list, list of macromolecule IDs in dataset
        graphs: list, list of graphs corresponding to each ID
        labels: list, list of labels corresponding to each ID
        masks: list, list of masks corresponding to each ID
        task: str, classification or regression
        n_tasks: int, number of tasks
        classtype: str, binary, multilabel, or multiclass for classification tasks
        normalizer: StandardScaler or QuantileTransformer for normalization
        
        '''
        self._df = pd.read_csv(DF_PATH)
        self._seed = SEED
        self._labelname = LABELNAME
        self._model = MODEL
        self._nx_graphs = NX_GRAPHS
        self._norm = NORM
        self.task = TASK
        self.normalizer = None
        self._convert_dgl()
    
    def _convert_dgl(self):
        ''' Utility function for conversion of featurized NetworkX to featurized DGL '''
        IDs_nonshuffle = self._df['ID'].tolist()
        np.random.seed(self._seed)
        indices = np.random.RandomState(seed=self._seed).permutation(np.arange(len(IDs_nonshuffle)))
        IDs_shuffle = list(np.array(IDs_nonshuffle)[indices])
        self.IDs = []
        graphs_feat = []
        for idnum in IDs_shuffle:
            if str(idnum) in self._nx_graphs.keys():
                self.IDs.append(idnum)
                graphs_feat.append(self._nx_graphs[str(idnum)])
            
        if self._model == 'GCN' or self._model == 'GAT':
            graphs_list = [dgl.from_networkx(graph_feat, node_attrs=['h'], edge_attrs=['e'], idtype=torch.int32) for graph_feat in graphs_feat]
            self.graphs = [dgl.add_self_loop(graph) for graph in graphs_list]
        else:
            self.graphs = [dgl.from_networkx(graph_feat, node_attrs=['h'], edge_attrs=['e'], idtype=torch.int32) for graph_feat in graphs_feat]
        
        if self.task == 'classification':
            self._classificationlabel()
        elif self.task == 'regression':
            self._regressionlabel()
        
    def _classificationlabel(self):
        ''' Utility function for assigning macromolecule labels for classification task '''
        label_list = self._getclass()
        labelname_list = self._df[self._labelname].tolist()
        comma_check = False
        for labelname in labelname_list:
            if not labelname is None and not pd.isna(labelname) and not labelname == '':
                if ',' in labelname:
                    comma_check = True
        if len(label_list) == 2 and comma_check == False:
            self.classtype = 'binary'
            self._binarylabelizer()
        elif len(label_list) > 2 and comma_check == False:
            self.classtype = 'multiclass'
            self._multilabelizer('mc')
        elif len(label_list) > 2 and comma_check == True:
            self.classtype = 'multilabel'
            self._multilabelizer('ml')
    
    def _binarylabelizer(self):
        ''' Utility function for assigning macromolecule labels and number of tasks for binary classification task '''
        labels = [] 
        masks = [] 
        label_list = self._getclass()
        for graph in self.IDs:
            label_str = self._df[self._df['ID']==graph][[self._labelname]].values[0].tolist()[0]
            mask_tmp = [1]
            if pd.isnull(label_str[0]): 
                mask_tmp[0] = 0
                label_str[0] = 0 
            label_tensor = torch.FloatTensor([label_list.index(label_str)])
            labels.append(label_tensor)
            mask_tensor = torch.FloatTensor(mask_tmp)
            masks.append(mask_tensor)
        
        self.labels = labels
        self.masks = masks
        self.n_tasks = 1
    
    def _multilabelizer(self, tasktype):
        ''' Utility function for assigning macromolecule labels and number of tasks for multilabel/multiclass classification task 
        
        Args:
        tasktype : str, 'ml' for multilabel or 'mc' for multiclass
        '''
        labels = [] 
        masks = [] 
        label_list = self._getclass()
        self.n_tasks = len(label_list)
        
        for graph in self.IDs:
            classname = self._df[self._df['ID']==graph][[self._labelname]].values[0].tolist()[0]
            if not classname is None and not pd.isna(classname) and not classname == '':
                itemlist = []
                if not classname.count(',') == 0:
                    itemlist += classname.strip().split(', ')
                else:
                    itemlist = [classname.strip()]
            
                label_str = [0 for a in range(len(label_list))]
                for item in itemlist:
                    nameindex = label_list.index(item.replace(',','').strip())
                    label_str[nameindex] = 1
                if tasktype == 'mc':
                    mask_tmp = [1]
                elif tasktype == 'ml':
                    mask_tmp = [1 for a in range(len(label_list))]
            else:
                label_str = [0 for a in range(len(label_list))]
                if tasktype == 'mc':
                    mask_tmp = [0]
                elif tasktype == 'ml':
                    mask_tmp = [0 for a in range(len(label_list))]
        
            label_tensor = torch.FloatTensor(label_str)
            mask_tensor = torch.FloatTensor(mask_tmp)
            labels.append(label_tensor)
            masks.append(mask_tensor)
            
        self.labels = labels
        self.masks = masks
        
    def _getclass(self):
        ''' Utility function for getting list of unique macromolecule labels 
        
        Returns:
        unique_list: list, list of unique macromolecule labels
        '''
        classlist = self._df[self._labelname].tolist()
        unique_list = []
        for classname in classlist:
            if not classname is None and not pd.isna(classname) and not classname == '':
                class_list = []
                if not classname.count(',') == 0:
                    class_list += classname.strip().split(', ')
                else:
                    class_list = [classname.strip()]
                for elem in class_list:
                    if elem.replace(',','').strip() not in unique_list:
                        unique_list.append(elem.replace(',','').strip())
        return unique_list
    
    def _regressionlabel(self):
        ''' Utility function for assigning macromolecule labels and number of tasks for regression task '''
        if self._norm == 'qt':
            self._quantiletransform()
        elif self._norm == 'ss':
            self._standardscaler()
        self.n_tasks = 1
        
    def _quantiletransform(self):
        ''' Utility function for normalizing regression labels using quantile transform '''
        df_list = self._df['avg'].tolist()
        data_list = [val for val in df_list if (not val is None and not pd.isna(val) and not pd.isnull(val) and not val == '')]
        qt = QuantileTransformer(n_quantiles = len(data_list), random_state = self._seed)
        qt.fit(np.array(data_list).reshape(-1, 1))
    
        labels = [] 
        masks = [] 
        for graph in self.IDs:
            label_orig = self._df[self._df['ID']==graph][['avg']].values[0].tolist()
            label_scale = list(qt.transform(np.array(label_orig).reshape(-1, 1)))[0]
            mask_tmp = [1]
            if pd.isnull(label_orig[0]): 
                mask_tmp[0] = 0
                label_scale[0] = 0 
            label_tensor = torch.FloatTensor(label_scale)
            labels.append(label_tensor)
            mask_tensor = torch.FloatTensor(mask_tmp)
            masks.append(mask_tensor) 
        self.labels = labels
        self.masks = masks
        self.normalizer = qt
        
    def _standardscaler(self):
        ''' Utility function for normalizing regression labels using standard scaler '''
        df_list = self._df['avg'].tolist()
        data_list = [val for val in df_list if (not val is None and not pd.isna(val) and not pd.isnull(val) and not val == '')]
        scaler = StandardScaler()
        scaler.fit(np.array(data_list).reshape(-1, 1))
    
        labels = [] 
        masks = [] 
        for graph in self.IDs:
            label_orig = self._df[self._df['ID']==graph][['avg']].values[0].tolist()
            label_scale = list(scaler.transform(np.array(label_orig).reshape(-1, 1)))[0]
            mask_tmp = [1]
            if pd.isnull(label_orig[0]): 
                mask_tmp[0] = 0
                label_scale[0] = 0 
            label_tensor = torch.FloatTensor(label_scale)
            labels.append(label_tensor)
            mask_tensor = torch.FloatTensor(mask_tmp)
            masks.append(mask_tensor) 
        self.labels = labels
        self.masks = masks
        self.normalizer = scaler
        
    def __getitem__(self, idx):
        '''Utility function for getting datapoint with index

        Args:
        idx : int, index of datapoint
        
        Returns:
        self.IDs[idx], self.graphs[idx], self.labels[idx], self.mask[idx]: ID, graph, label, mask of specified index
        '''
        return self.IDs[idx], self.graphs[idx], self.labels[idx], self.masks[idx]
    
    def __len__(self):
        '''Utility function to find number of graphs in the dataset
        
        Returns:
        len(self.graphs): int, number of graphs in dataset
        '''
        return len(self.graphs)
    
