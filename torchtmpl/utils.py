# coding: utf-8

# Standard imports
import os
import logging
# External imports
import torch
import torch.nn
import tqdm
import numpy as np
import pandas as pd


def generate_unique_logpath(logdir, raw_run_name):
    """
    Generate a unique directory name
    Argument:
        logdir: the prefix directory
        raw_run_name(str): the base name
    Returns:
        log_path: a non-existent path like logdir/raw_run_name_xxxx
                  where xxxx is an int
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


class ModelCheckpoint(object):
    """
    Early stopping callback
    """

    def __init__(
        self,
        model: torch.nn.Module,
        savepath,
        min_is_best: bool = True,
    ) -> None:
        self.model = model
        self.savepath = savepath
        self.best_score = None
        if min_is_best:
            self.is_better = self.lower_is_better
        else:
            self.is_better = self.higher_is_better

    def lower_is_better(self, score):
        return self.best_score is None or score < self.best_score

    def higher_is_better(self, score):
        return self.best_score is None or score > self.best_score

    def update(self, score):
        if self.is_better(score):
            torch.save(self.model.state_dict(), self.savepath)
            self.best_score = score
            return True
        return False


def train(model, loader, f_loss, optimizer, device, dynamic_display=True):
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    f_loss    -- The loss function, i.e. a loss Module
    optimizer -- A torch.optim.Optimzer object
    device    -- A torch.device
    Returns :
    The averaged train metrics computed over a sliding window
    """

    # We enter train mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.train()

    total_loss = 0
    num_samples = 0
    for i, data in (pbar := tqdm.tqdm(enumerate(loader))):
        input_imgs = data[0].to(device)
        targets = data[-1].to(device)
        if len(data) == 3:
            input_tab = data[1].to(device)
        optimizer.zero_grad()
        if len(data) == 2:
            outputs = model(input_imgs)
        elif len(data) == 3:
            outputs = model(input_imgs, input_tab)
        loss = f_loss(outputs, targets)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Update the metrics
        # We here consider the loss is batch normalized
        total_loss += input_imgs.shape[0] * loss.item()
        num_samples += input_imgs.shape[0]
        pbar.set_description(f"Train loss : {total_loss/num_samples:.2f}")
    return total_loss / num_samples


def test(model, loader, f_loss, device):
    """
    Test a model over the loader
    using the f_loss as metrics
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    f_loss    -- The loss function, i.e. a loss Module
    device    -- A torch.device
    Returns :
    """

    # We enter eval mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.eval()

    total_loss = 0
    num_samples = 0
    total_topk_accuracy = 0
    
    for i, data in (pbar := tqdm.tqdm(enumerate(loader))):

        input_imgs = data[0].to(device)
        targets = data[-1].to(device)
        if len(data) == 3:
            input_tab = data[1].to(device)
        if len(data) == 2:
            outputs = model(input_imgs)
        elif len(data) == 3:
            outputs = model(input_imgs, input_tab)
        loss = f_loss(outputs, targets)

        # Update the metrics
        total_topk_accuracy += input_imgs.shape[0] * topk_accuracy(outputs, targets).item()

        # We here consider the loss is batch normalized
        total_loss += input_imgs.shape[0] * loss.item()
        num_samples += input_imgs.shape[0]

    return total_loss / num_samples,1 - (total_topk_accuracy / num_samples)


def topk_accuracy(outputs, targets, k=30):
    """
    Compute the top-k accuracy given the model outputs and target labels.
    Arguments:
    outputs -- Model predictions
    targets -- Target labels
    k       -- The top-k parameter for accuracy calculation (default is 1)
    Returns:
    topk_acc -- Top-k accuracy
    """
    _, indices = torch.topk(outputs, k, sorted=False)
    targets = targets.view(-1, 1).expand_as(indices)
    correct = indices.eq(targets)
    topk_acc = (correct.float().sum() )/ targets.size(0)
    return topk_acc


def predict(model, loader, device):
    """
    Test a model over the loader
    using the f_loss as metrics
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    f_loss    -- The loss function, i.e. a loss Module
    device    -- A torch.device
    Returns :
    """

    # We enter eval mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.eval()
    
    prediction = []

    for i, data in (pbar := tqdm.tqdm(enumerate(loader))):
        if len(data) == 2:
            input_imgs = data[0].to(device)
            input_tab = data[1].to(device)
            outputs = model(input_imgs, input_tab)
        else:
            outputs = model(data.to(device))
        # Get the 30 top predictions
        _, indices = torch.topk(outputs, 30, sorted=True)
        prediction.extend(indices.cpu().numpy())
    return prediction


def save_predictions_to_csv(predictions, ids, submitdir):
    """
    Save predictions to a CSV file.

    Args:
    - predictions (list of numpy arrays): List of predictions.
    - ids (numpy array): Array of IDs corresponding to the predictions.
    - submitdir (pathlib.Path): Path to the directory where the submission file will be saved.
    """
    logging.info("= Saving the prediction")

    s_pred = [" ".join(map(str, pred_set)) for pred_set in predictions]

    df = pd.DataFrame(
        {
            "Id": ids,
            "Predicted": s_pred,
        }
    )
    df.to_csv(os.path.join(submitdir,"submission.csv"), index=False)




    
    

