import openke
from openke.config import Trainer, Tester
from openke.module.loss import SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

from openke_model import RotAtte

import torch
import numpy as np

import optuna
import neptune
import neptunecontrib.monitoring.optuna as opt_utils

from config import API_TOKEN as API_TOKEN


def objective(trial: optuna.Trial):
    negative_rate = trial.suggest_categorical("negative_rate", [10, 20, 30, 40, 50, 60])
    in_dim = trial.suggest_categorical('input dim', [100, 200, 500, 1000])
    out_dim = in_dim 
    alpha = trial.suggest_loguniform('alpha', 2e-6, 2e-1)
    return run_experiment(negative_rate, in_dim, out_dim, alpha)

def run_experiment(negative_rate, in_dim, out_dim, alpha):
    train_dataloader = TrainDataLoader(
        in_path = "./benchmarks/FB15K237/", 
        batch_size = 10000,
        threads = 1,
        sampling_mode = "cross", 
        bern_flag = 0, 
        filter_flag = 1, 
        neg_ent = negative_rate,
        neg_rel = 0
    )

    facts = TrainDataLoader(
        in_path = "./benchmarks/FB15K237/", 
        batch_size = train_dataloader.get_triple_tot(),
        threads = 1,
        sampling_mode = "normal", 
        bern_flag = 0, 
        filter_flag = 1, 
        neg_ent = 0,
        neg_rel = 0
    )

    h, t, r, _, _ = [f for f in facts][0].values()
    h = torch.Tensor(h).to(torch.long)
    t = torch.Tensor(t).to(torch.long)
    r = torch.Tensor(r).to(torch.long)

    facts = torch.stack((h, r, t)).cuda().t()

    test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

    rotatte = RotAtte(
        n_ent = train_dataloader.get_ent_tot(),
        n_rel = train_dataloader.get_rel_tot(),
        in_dim = in_dim, 
        out_dim = in_dim,
        facts = facts,
        negative_rate=negative_rate,
    )

    model = NegativeSampling(
        model = rotatte, 
        loss = SigmoidLoss(adv_temperature = 2),
        batch_size = train_dataloader.get_batch_size(), 
        regul_rate = 0.0
    )

    trainer = Trainer(model = model, data_loader = train_dataloader, 
        train_times = 100, alpha = alpha, use_gpu = True, opt_method = "adam")

    trainer.run()
    tester = Tester(model = rotatte, data_loader = test_dataloader, use_gpu = True)
    result = tester.run_link_prediction(type_constrain = False)

    MRR, MR, hits10, hits3, hits1 = result

    return MRR 


neptune.init("saiashirwad/relation-prediction", api_token=API_TOKEN)
neptune.create_experiment(name="optuna sweep")

monitor = opt_utils.NeptuneMonitor()
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)