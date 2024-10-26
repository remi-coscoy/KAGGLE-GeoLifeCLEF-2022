# coding: utf-8

# Standard imports
import logging
import sys
import os
import pathlib
import numpy as np

# External imports
import yaml
import wandb
import torch
import torchinfo.torchinfo as torchinfo

# Local imports
from torchtmpl import data,models,optim,utils


def train(config):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    if "wandb" in config["logging"]:
        wandb_config = config["logging"]["wandb"]
        wandb.login(key=wandb_config["key"])
        wandb.init(project=wandb_config["project"], entity=wandb_config["entity"])
        wandb_log = wandb.log
        wandb_log(config)
        logging.info("Will be recording in wandb run name : {wandb.run.name}")
    else:
        wandb_log = None

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]

    train_loader, valid_loader,test_loader, input_img_size, input_tab_size, num_classes, ids =data.get_dataloaders(
        data_config, use_cuda
    )

    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_img_size, input_tab_size, num_classes)
    model.to(device)

    # Build the loss
    logging.info("= Loss")
    loss = optim.get_loss(config["loss"])

    # Build the optimizer
    logging.info("= Optimizer")
    optim_config = config["optim"]
    optimizer = optim.get_optimizer(optim_config, model.parameters())

    # Build the callbacks
    logging_config = config["logging"]
    # Let us use as base logname the class name of the modek
    logname = model_config["class"]
    logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")

    # Copy the config file into the logdir
    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as file:
        yaml.dump(config, file)

    # Make a summary script of the experiment
    input_img_size = next(iter(train_loader))[0].shape
    input_tab_size = next(iter(train_loader))[1].shape
    logging.info(f"Input image size : {input_img_size} and input tab size : {input_tab_size}")
    summary_text = (
        f"Logdir : {logdir}\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + f" Config : {config} \n\n"
        + (f" Wandb run name : {wandb.run.name}\n\n" if wandb_log is not None else "")
        + "## Summary of the model architecture\n"
        + f"{torchinfo.summary(model, input_img_size=input_img_size,input_tab_size=input_tab_size)}\n\n"
        + "## Loss\n\n"
        + f"{loss}\n\n"
        + "## Datasets : \n"
        + f"Train : {train_loader.dataset}\n"
        + f"Validation : {valid_loader.dataset}"
    )

    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(summary_text)
    if wandb_log is not None:
        wandb.log({"summary": summary_text})

    # Define the early stopping callback
    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=True
    )

    for e in range(config["nepochs"]):
        # Train 1 epoch
        train_loss = utils.train(model, train_loader, loss, optimizer, device)

        # Test
        test_loss , test_metric = utils.test(model, valid_loader, loss, device)

        updated = model_checkpoint.update(test_loss)
        logging.info(
            "[%d/%d] Test loss : %.3f %s"
            % (
                e+1,
                config["nepochs"],
                test_loss,
                "[>> BETTER <<]" if updated else "",
            )
        )

        logging.info(
            f"Topk Error :{test_metric}" 
        )

        # Update the dashboard
        metrics = {"train_CE": train_loss, "test_CE": test_loss, "Top30error": test_metric}
        if wandb_log is not None:
            logging.info("Logging on wandb")
            wandb_log(metrics)

        # Make a prediction : saving a submission for each epoch
        logging.info("= Making a prediction")
        predictions = utils.predict(model, test_loader, device)
        logging.info("= Saving the prediction")
        submitdir = os.path.join(logdir,f"submit_epoch_{e}")
        os.makedirs(submitdir,exist_ok=True)
        utils.save_predictions_to_csv(predictions,ids,submitdir)
        
        logging.info("= Done")
        


def test(config,model_number):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    
    # Build the dataloaders
    logging.info("= Building the test dataloaders")
    data_config = config["data"]
    
    logging_config = config["logging"]
    logdir = os.path.join(logging_config["logdir"], model_number)
    if not os.path.isdir(logdir):
        logging.error(f"Model {model_number} does not exist")
        sys.exit(-1)
    logging.info(f"Will be loading model {model_number} from {logdir}")
    
    test_loader, input_img_size, input_tab_size,  num_classes,ids = data.get_test_dataloader(data_config, use_cuda)
    # load the model : best_model.pt
    model_config = config["model"]
    model = models.build_model(model_config, input_img_size, input_tab_size, num_classes)
    model.to(device)
    log_path = os.path.join(logdir, "best_model.pt")
    model.load_state_dict(torch.load(log_path))
    
    # submit dir
    submit_config = config["submit"]
    # Let us use as base logname the class name of the modek
    submitdir = os.path.join(submit_config["submitdir"], model_number)
    if not os.path.isdir(submitdir):
        os.makedirs(submitdir)
    logging.info(f"Will be submitting into {submitdir}")

    submitdir = pathlib.Path(submitdir)
    with open(submitdir / "config.yaml", "w") as file:
        yaml.dump(config, file)
    
    # Make a prediction
    logging.info("= Making a prediction")
    predictions = utils.predict(model, test_loader, device)
    logging.info("= Saving the prediction")

    utils.save_predictions_to_csv(predictions,ids,submitdir)
    
    logging.info("= Done")



if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    if len(sys.argv) < 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test> <model_number if test>")
        sys.exit(-1)
    if len(sys.argv) == 3:
        if sys.argv[2] != "train":
            logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test> <model_number if test>")
            sys.exit(-1)
        else:
            logging.info("Loading {}".format(sys.argv[1]))
            config = yaml.safe_load(open(sys.argv[1], "r"))

            command = sys.argv[2]
            train(config)
    
    
    if len(sys.argv) == 4:
        if sys.argv[2] != "test":
            logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test> <model_number if test>")
            sys.exit(-1)
        else:
            logging.info("Loading {}".format(sys.argv[1]))
            config = yaml.safe_load(open(sys.argv[1], "r"))

            command = sys.argv[2]
            model_number = sys.argv[3]
            test(config,model_number)

