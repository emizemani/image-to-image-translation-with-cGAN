class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        """
        Initialize early stopping.
        Args:
            patience (int): Number of epochs to wait before stopping after loss plateaus
            min_delta (float): Minimum change in loss to qualify as an improvement
            verbose (bool): If True, prints message when saving checkpoint
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0 