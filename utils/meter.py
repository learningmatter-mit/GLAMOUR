import torch

import numpy as np
import torch.nn.functional as F

from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, auc, hamming_loss, roc_curve


class Meter_v2():
    def __init__(self, mean=None, std=None):
        '''
        Initializes a Meter_v2 object
        
        Args:
        mean : torch.float32 tensor of shape (T) or None, mean of existing training labels across tasks
        std : torch.float32 tensor of shape (T) or None, std of existing training labels across tasks
        
        '''
        self._mask = []
        self.y_pred = []
        self.y_true = []

        if (mean is not None) and (std is not None):
            self._mean = mean.cpu()
            self._std = std.cpu()
        else:
            self._mean = None
            self._std = None

    def update(self, y_pred, y_true, mask=None):
        '''Updates for the result of an iteration

        Args:
        y_pred : float32 tensor, predicted labels with shape (B, T), B for number of graphs in the batch and T for number of tasks
        y_true : float32 tensor, ground truth labels with shape (B, T), B for number of graphs in the batch and T for number of tasks
        mask : None or float32 tensor, binary mask indicating the existence of ground truth labels
        '''
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        if mask is None:
            self._mask.append(torch.ones(self.y_pred[-1].shape))
        else:
            self._mask.append(mask.detach().cpu())

    def _finalize(self):
        '''Utility function for preparing for evaluation.

        Returns:
        mask : float32 tensor, binary mask indicating the existence of ground truth labels
        y_pred : float32 tensor, predicted labels with shape (B, T), B for number of graphs in the batch and T for number of tasks
        y_true : float32 tensor, ground truth labels with shape (B, T), B for number of graphs in the batch and T for number of tasks
        '''
        mask = torch.cat(self._mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)

        if (self._mean is not None) and (self._std is not None):
            y_pred = y_pred * self._std + self._mean

        return mask, y_pred, y_true

    def _reduce_scores(self, scores, reduction='none'):
        '''Utility function for finalizing the scores to return.

        Args:
        scores : list, list of scores for all tasks.
        reduction : str, 'none' or 'mean' or 'sum' to control the form of scores for all tasks

        Returns:
        float or list of float, depending on reduction type
            * If reduction == 'none', list of scores for all tasks.
            * If reduction == 'mean', mean of scores for all tasks.
            * If reduction == 'sum', sum of scores for all tasks.
        '''
        if reduction == 'none':
            return scores
        elif reduction == 'mean':
            return np.mean(scores)
        elif reduction == 'sum':
            return np.sum(scores)
        else:
            raise ValueError(
                "Expect reduction to be 'none', 'mean' or 'sum', got {}".format(reduction))

    def multilabel_score(self, score_func, reduction='none'):
        '''Evaluate for multi-label prediction.

        ArgsL
        score_func : callable function, score function that takes task-specific ground truth and predicted labels as
            input and return a float as the score
        reduction : str, 'none' or 'mean' or 'sum' to control the form of scores for all tasks

        Returns:
        float or list of float, depending on reduction type
            * If reduction == 'none', list of scores for all tasks.
            * If reduction == 'mean', mean of scores for all tasks.
            * If reduction == 'sum', sum of scores for all tasks.
        '''
        mask, y_pred, y_true = self._finalize()
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            task_score = score_func(task_y_true, task_y_pred)
            if task_score is not None:
                scores.append(task_score)
        return self._reduce_scores(scores, reduction)

    def pearson_r2(self, reduction='none'):
        '''Compute squared Pearson correlation coefficient.

        Args:
        reduction : str, 'none' or 'mean' or 'sum' to control the form of scores for all tasks

        Returns:
        float or list of float, depending on reduction type
            * If reduction == 'none', list of scores for all tasks.
            * If reduction == 'mean', mean of scores for all tasks.
            * If reduction == 'sum', sum of scores for all tasks.
        '''
        def score(y_true, y_pred):
            return pearsonr(y_true.numpy(), y_pred.numpy())[0] ** 2
        return self.multilabel_score(score, reduction)

    def mae(self, reduction='none'):
        '''Compute mean absolute error.

        Args:
        reduction : str, 'none' or 'mean' or 'sum' to control the form of scores for all tasks

        Returns:
        float or list of float, depending on reduction type
            * If reduction == 'none', list of scores for all tasks.
            * If reduction == 'mean', mean of scores for all tasks.
            * If reduction == 'sum', sum of scores for all tasks.
        '''
        def score(y_true, y_pred):
            return F.l1_loss(y_true, y_pred).data.item()
        return self.multilabel_score(score, reduction)

    def rmse(self, reduction='none'):
        '''Compute root mean square error.

        Args:
        reduction : str, 'none' or 'mean' or 'sum' to control the form of scores for all tasks

        Returns:
        float or list of float, depending on reduction type
            * If reduction == 'none', list of scores for all tasks.
            * If reduction == 'mean', mean of scores for all tasks.
            * If reduction == 'sum', sum of scores for all tasks.
        '''
        def score(y_true, y_pred):
            return torch.sqrt(F.mse_loss(y_pred, y_true).cpu()).item()
        return self.multilabel_score(score, reduction)
    
    def spearmanr(self, reduction='none'):
        '''Compute Spearman correlation coefficient.

        Args:
        reduction : str, 'none' or 'mean' or 'sum' to control the form of scores for all tasks

        Returns:
        float or list of float, depending on reduction type
            * If reduction == 'none', list of scores for all tasks.
            * If reduction == 'mean', mean of scores for all tasks.
            * If reduction == 'sum', sum of scores for all tasks.
        '''
        def score(y_true, y_pred):
            return spearmanr(y_true.numpy(), y_pred.numpy())[0]
        return self.multilabel_score(score, reduction)
    
    def kendalltau(self, reduction='none'):
        '''Compute Kendall's tau.

        Args:
        reduction : str, 'none' or 'mean' or 'sum' to control the form of scores for all tasks

        Returns:
        float or list of float, depending on reduction type
            * If reduction == 'none', list of scores for all tasks.
            * If reduction == 'mean', mean of scores for all tasks.
            * If reduction == 'sum', sum of scores for all tasks.
        '''
        def score(y_true, y_pred):
            return kendalltau(y_true.numpy(), y_pred.numpy())[0]
        return self.multilabel_score(score, reduction)

    def roc_auc_score(self, reduction='none'):
        '''Compute the area under the receiver operating characteristic curve (roc-auc score)
        for binary classification.

        ROC-AUC scores are not well-defined in cases where labels for a task have one single
        class only (e.g. positive labels only or negative labels only). In this case we will
        simply ignore this task and print a warning message.

        Args:
        reduction : str, 'none' or 'mean' or 'sum' to control the form of scores for all tasks

        Returns:
        float or list of float, depending on reduction type
            * If reduction == 'none', list of scores for all tasks.
            * If reduction == 'mean', mean of scores for all tasks.
            * If reduction == 'sum', sum of scores for all tasks.
        '''
        assert (self._mean is None) and (self._std is None),             'Label normalization should not be performed for binary classification.'
        
        mask, y_pred, y_true = self._finalize()
        y_pred_numpy = torch.FloatTensor(torch.sigmoid(y_pred).numpy()) * (mask != 0).float()
        return roc_auc_score(y_true.long().numpy(), y_pred_numpy.numpy(), average = 'micro')

    def pr_auc_score(self, reduction='none'):
        '''Compute the area under the precision-recall curve (pr-auc score)
        for binary classification.

        PR-AUC scores are not well-defined in cases where labels for a task have one single
        class only (e.g. positive labels only or negative labels only). In this case, we will
        simply ignore this task and print a warning message.

        Args:
        reduction : str, 'none' or 'mean' or 'sum' to control the form of scores for all tasks

        Returns:
        float or list of float, depending on reduction type
            * If reduction == 'none', list of scores for all tasks.
            * If reduction == 'mean', mean of scores for all tasks.
            * If reduction == 'sum', sum of scores for all tasks.
        '''
        assert (self._mean is None) and (self._std is None),             'Label normalization should not be performed for binary classification.'
        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print('Warning: Only one class {} present in y_true for a task. '
                      'PR AUC score is not defined in that case.'.format(y_true[0]))
                return None
            else:
                precision, recall, _ = precision_recall_curve(
                    y_true.long().numpy(), torch.sigmoid(y_pred).numpy())
                return auc(recall, precision)
        return self.multilabel_score(score, reduction)
    
    def f1_score(self, reduction='none'):
        '''Compute the weighted average of the precision and recall, where an F1 score reaches
        its best value at 1 and worst value at 0

        Args:
        reduction : str, 'none' or 'mean' or 'sum' to control the form of scores for all tasks

        Returns:
        float or list of float, depending on reduction type
            * If reduction == 'none', list of scores for all tasks.
            * If reduction == 'mean', mean of scores for all tasks.
            * If reduction == 'sum', sum of scores for all tasks.
        '''
        mask, y_pred, y_true = self._finalize()
        y_pred_numpy = torch.sigmoid(y_pred).numpy()
        y_pred_round = torch.FloatTensor(np.around(y_pred_numpy)) * (mask != 0).float()
        return f1_score(y_true.long().numpy(), y_pred_round.numpy(), average = 'micro')

    def precision_score(self, reduction='none'):
        '''Compute the ratio of the true positives to the sum of true and false positives. The precision is 
        intuitively the ability of the classifier not to label as positive a sample that is negative.

        Args:
        reduction : str, 'none' or 'mean' or 'sum' to control the form of scores for all tasks

        Returns:
        float or list of float, depending on reduction type
            * If reduction == 'none', list of scores for all tasks.
            * If reduction == 'mean', mean of scores for all tasks.
            * If reduction == 'sum', sum of scores for all tasks.
        '''
        mask, y_pred, y_true = self._finalize()
        y_pred_numpy = torch.sigmoid(y_pred).numpy()
        y_pred_round = torch.FloatTensor(np.around(y_pred_numpy)) * (mask != 0).float()
        return precision_score(y_true.long().numpy(), y_pred_round.numpy(), average = 'micro', zero_division = 1)
    
    def recall_score(self, reduction='none'):
        '''Compute the ratio of the true positives to the sum of true positives and false negatives. The recall is 
        intuitively the ability of the classifier not to find all the positive samples. 

        Args:
        reduction : str, 'none' or 'mean' or 'sum' to control the form of scores for all tasks

        Returns:
        float or list of float, depending on reduction type
            * If reduction == 'none', list of scores for all tasks.
            * If reduction == 'mean', mean of scores for all tasks.
            * If reduction == 'sum', sum of scores for all tasks.
        '''
        mask, y_pred, y_true = self._finalize()
        y_pred_numpy = torch.sigmoid(y_pred).numpy()
        y_pred_round = torch.FloatTensor(np.around(y_pred_numpy)) * (mask != 0).float()
        return recall_score(y_true.long().numpy(), y_pred_round.numpy(), average = 'micro')
    
    def accuracy_score(self, reduction='none'):
        '''Compute the accuracy classification score, or the fraction of correctly classified samples

        Args:
        reduction : str, 'none' or 'mean' or 'sum' to control the form of scores for all tasks

        Returns:
        float or list of float, depending on reduction type
            * If reduction == 'none', list of scores for all tasks.
            * If reduction == 'mean', mean of scores for all tasks.
            * If reduction == 'sum', sum of scores for all tasks.
        '''
        mask, y_pred, y_true = self._finalize()
        y_pred_numpy = torch.sigmoid(y_pred).numpy()
        y_pred_round = torch.FloatTensor(np.around(y_pred_numpy)) * (mask != 0).float()
        return accuracy_score(y_true.long().numpy(), y_pred_round.numpy())
    
    def hamming_loss(self, reduction='none'):
        '''Compute the fraction of labels that are incorrectly predicted

        Args:
        reduction : str, 'none' or 'mean' or 'sum' to control the form of scores for all tasks

        Returns:
        float or list of float, depending on reduction type
            * If reduction == 'none', list of scores for all tasks.
            * If reduction == 'mean', mean of scores for all tasks.
            * If reduction == 'sum', sum of scores for all tasks.
        '''
        mask, y_pred, y_true = self._finalize()
        y_pred_numpy = torch.sigmoid(y_pred).numpy()
        y_pred_round = torch.FloatTensor(np.around(y_pred_numpy)) * (mask != 0).float()
        return hamming_loss(y_true.long().numpy(), y_pred_round.numpy())
    
    def roc_curve(self, reduction='none'):
        '''Compute Receiver operating characteristic (ROC)

        Args:
        reduction : str, 'none' or 'mean' or 'sum' to control the form of scores for all tasks

        Returns:
        float or list of float, depending on reduction type
            * If reduction == 'none', list of scores for all tasks.
            * If reduction == 'mean', mean of scores for all tasks.
            * If reduction == 'sum', sum of scores for all tasks.
        '''
        mask, y_pred, y_true = self._finalize()
        y_pred_numpy = torch.FloatTensor(torch.sigmoid(y_pred).numpy()) * (mask != 0).float()
        fpr, tpr, _ = roc_curve(y_true.long().numpy().ravel(), y_pred_numpy.numpy().ravel())
        return [fpr, tpr]
    
    def inverse(self, normalizer):
        pred_list = []
        true_list = []
            
        for index in range(len(self.y_pred)):
            pred_list.append(torch.FloatTensor(normalizer.inverse_transform(self.y_pred[index])))
            true_list.append(torch.FloatTensor(normalizer.inverse_transform(self.y_true[index])))
                
        true_tensor = torch.cat(true_list, dim=0)
        pred_tensor = torch.cat(pred_list, dim=0)
            
        return [list(true_tensor), list(pred_tensor)]
    
    def compute_metric(self, metric_name, reduction='none'):
        '''Compute metric based on metric name.

        Args:
        metric_name : str

            * 'r2': compute squared Pearson correlation coefficient
            * 'mae': compute mean absolute error
            * 'rmse': compute root mean square error
            * 'spearmanr': compute Spearman's rank correlation coefficient
            * 'kendalltau': compute Kendall's Tau
            * 'roc_auc_score': compute roc-auc score
            * 'pr_auc_score': compute pr-auc score
            * 'f1 score': compute f1 score
            * 'precision_score': compute precision score
            * 'recall_score': compute recall score
            * 'accuracy_score': compute accuracy score
            * 'hamming_loss': compute hamming loss

        reduction : str, 'none' or 'mean' or 'sum' to control the form of scores for all tasks

        Returns:
        float or list of float, depending on reduction type
            * If reduction == 'none', list of scores for all tasks.
            * If reduction == 'mean', mean of scores for all tasks.
            * If reduction == 'sum', sum of scores for all tasks.
        '''
        if metric_name == 'r2':
            return self.pearson_r2(reduction)
        elif metric_name == 'mae':
            return self.mae(reduction)
        elif metric_name == 'rmse':
            return self.rmse(reduction)
        elif metric_name == 'spearmanr':
            return self.spearmanr(reduction)
        elif metric_name == 'kendalltau':
            return self.kendalltau(reduction)
        elif metric_name == 'roc_auc_score':
            return self.roc_auc_score(reduction)
        elif metric_name == 'pr_auc_score':
            return self.pr_auc_score(reduction)
        elif metric_name == 'f1_score':
            return self.f1_score('mean')
        elif metric_name == 'precision_score':
            return self.precision_score('mean')
        elif metric_name == 'recall_score':
            return self.recall_score('mean')
        elif metric_name == 'accuracy_score':
            return self.accuracy_score(reduction)
        elif metric_name == 'hamming_loss':
            return self.hamming_loss(reduction)
        elif metric_name == 'roc_curve':
            return self.roc_curve(reduction)
