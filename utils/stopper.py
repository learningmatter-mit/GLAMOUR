import torch

from dgllife.utils import EarlyStopping


class Stopper_v2(object):
    def __init__(self, savepath, mode='higher', patience=10, filename=None, metric=None):
        '''
        Initializes a Stopper_v2 object
        
        Args:
        mode : str, 'higher' if higher metric suggests a better model or 'lower' if lower metric suggests a better model
        patience : int, number of consecutive epochs with no observed performance required for early stopping
        filename : str or None, filename for storing the model checkpoint. 
        If not specified, we will automatically generate a file starting with 'early_stop'
        based on the current time.
        metric : str or None, metric name
        '''
        if filename is None:
            dt = datetime.datetime.now()
            filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
                dt.date(), dt.hour, dt.minute, dt.second)

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self._patience = patience
        self._counter = 0
        self._savepath = savepath
        self._filename = filename
        self.best_score = None
        self._early_stop = False

    def _check_higher(self, score, prev_best_score):
        '''Utility function for checking if the new score is higher than the previous best score.

        Args:
        score : float, new score
        prev_best_score : float, previous best score

        Returns:
        bool : boolean, whether the new score is higher than the previous best score.
        '''
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        '''Utility function for checking if the new score is lower than the previous best score.

        Args:
        score : float, new score
        prev_best_score : float, previous best score

        Returns:
        bool : boolearn, whether the new score is lower than the previous best score.
        '''
        return score < prev_best_score

    def step(self, score, model, optimizer, modelname, save_model, save_opt):
        '''Updates based on a new score, which is typically model performance on the validation set
        for a new epoch.

        Args:
        score : float, new score
        model : nn.Module, model instance
        optimizer : torch.optim.Adam, Adam instance
        modelname : str, name of model architecture
        save_model : boolean, whether to save full model
        save_opt : boolean, whether to save optimizer

        Returns:
        self._early_stop: boolean, whether an early stop should be performed.
        '''
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, optimizer, modelname, save_model, save_opt)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model, optimizer, modelname, save_model, save_opt)
            self._counter = 0
        else:
            self._counter += 1
            print(
                f'EarlyStopping counter: {self._counter} out of {self._patience}')
            if self._counter >= self._patience:
                self._early_stop = True
        return self._early_stop


    def save_checkpoint(self, model, optimizer, modelname, save_model, save_opt):
        '''Saves model when the metric on the validation set gets improved.

        Args:
        model : nn.Module, model instance
        optimizer : torch.optim.Adam, Adam instance
        modelname : str, name of model architecture
        save_model : boolean, whether to save full model
        save_opt : boolean, whether to save optimizer
        '''
        torch.save({'model_state_dict': model.state_dict()}, self._filename)
        
        if save_model == True:
            if not modelname == 'MPNN':
                torch.save(model, self._savepath + '/fullmodel.pt')
                
        if save_opt == True:
            torch.save(optimizer, self._savepath + '/fulloptimizer.pt')


    def load_checkpoint(self, model):
        '''Load the latest checkpoint

        Args:
        model : nn.Module, model instance
        '''
        model.load_state_dict(torch.load(self._filename)['model_state_dict'])



