import torch
import numpy as np
import matplotlib
from torch import nn
import time
from scgpt.model import TransformerGenerator
from typing import Iterable, List, Tuple, Dict, Union, Optional
import warnings
from gears.utils import create_cell_graph_dataset_for_prediction
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from tqdm import tqdm


def train(model: nn.Module, train_loader: torch.utils.data.DataLoader,
          device,
          include_zero_gene,
          n_genes,
          max_seq_len,
          map_raw_id_to_vocab_id,
          gene_ids,
          amp,
          CLS,
          CCE,
          MVC,
          ECS,
          criterion,
          scaler,
          optimizer,
          logger,
          log_interval,
          scheduler,
          epoch) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, total_mse = 0.0, 0.0
    start_time = time.time()

    num_batches = len(train_loader)
    for batch, batch_data in enumerate(train_loader):
        batch_size = len(batch_data.y)
        batch_data.to(device)
        x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
        # ori_gene_values = x[:, 0].view(batch_size, n_genes)
        ori_gene_values = x
        # pert_flags = x[:, 1].long().view(batch_size, n_genes)
        pert_flags = batch_data.pert_flags.long()
        target_gene_values = batch_data.y  # (batch_size, n_genes)

        if include_zero_gene in ["all", "batch-wise"]:
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            else:
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            # sample input_gene_id
            if len(input_gene_ids) > max_seq_len:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                    :max_seq_len
                ]
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )

        with torch.cuda.amp.autocast(enabled=amp):
            output_dict = model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
            )
            output_values = output_dict["mlm_output"]

            masked_positions = torch.ones_like(
                input_values, dtype=torch.bool
            )  # Use all
            loss = loss_mse = criterion(output_values, target_values, masked_positions)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        # torch.cuda.empty_cache()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.5f} | mse {cur_mse:5.5f} |"
            )
            total_loss = 0
            total_mse = 0
            start_time = time.time()


def evaluate(model: nn.Module, val_loader: torch.utils.data.DataLoader,
             device,
            include_zero_gene,
            n_genes,
            max_seq_len,
            map_raw_id_to_vocab_id,
            gene_ids,
            amp,
            CLS,
            CCE,
            MVC,
            ECS,
            criterion,
            masked_relative_error,
            scaler,
            optimizer,
            warnings,
            logger,
            log_interval,
            scheduler,
            epoch) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0

    with torch.no_grad():
        for batch, batch_data in enumerate(val_loader):
            batch_size = len(batch_data.y)
            batch_data.to(device)
            x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
            # ori_gene_values = x[:, 0].view(batch_size, n_genes)
            ori_gene_values = x
            # pert_flags = x[:, 1].long().view(batch_size, n_genes)
            pert_flags = batch_data.pert_flags.long()
            target_gene_values = batch_data.y  # (batch_size, n_genes)

            if include_zero_gene in ["all", "batch-wise"]:
                if include_zero_gene == "all":
                    input_gene_ids = torch.arange(n_genes, device=device)
                else:  # when batch-wise
                    input_gene_ids = (
                        ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                    )

                # sample input_gene_id
                if len(input_gene_ids) > max_seq_len:
                    input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                        :max_seq_len
                    ]
                input_values = ori_gene_values[:, input_gene_ids]
                input_pert_flags = pert_flags[:, input_gene_ids]
                target_values = target_gene_values[:, input_gene_ids]

                mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
                mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
                src_key_padding_mask = torch.zeros_like(
                    input_values, dtype=torch.bool, device=input_values.device
                )
            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = model(
                    mapped_input_gene_ids,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=CLS,
                    CCE=CCE,
                    MVC=MVC,
                    ECS=ECS,
                    do_sample=True,
                )
                output_values = output_dict["mlm_output"]

                masked_positions = torch.ones_like(
                    input_values, dtype=torch.bool, device=input_values.device
                )
                loss = criterion(output_values, target_values, masked_positions)
            total_loss += loss.item()
            total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item()
    return total_loss / len(val_loader), total_error / len(val_loader)

def predict(
    model: TransformerGenerator, pert_list: List[str], pool_size: Optional[int] = None,
    pert_data = None,
    eval_batch_size = None,
    include_zero_gene = None,
    gene_ids = None,
    amp = None,
) -> Dict:
    """
    Predict the gene expression values for the given perturbations.

    Args:
        model (:class:`torch.nn.Module`): The model to use for prediction.
        pert_list (:obj:`List[str]`): The list of perturbations to predict.
        pool_size (:obj:`int`, optional): For each perturbation, use this number
            of cells in the control and predict their perturbation results. Report
            the stats of these predictions. If `None`, use all control cells.
    """
    adata = pert_data.adata
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
    if pool_size is None:
        pool_size = len(ctrl_adata.obs)
    gene_list = pert_data.gene_names.values.tolist()
    for pert in pert_list:
        for i in pert:
            if i not in gene_list:
                raise ValueError(
                    "The gene is not in the perturbation graph. Please select from GEARS.gene_list!"
                )

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        results_pred = {}
        for pert in pert_list:
            cell_graphs = create_cell_graph_dataset_for_prediction(
                pert, ctrl_adata, gene_list, device, num_samples=pool_size
            )
            loader = DataLoader(cell_graphs, batch_size=eval_batch_size, shuffle=False)
            preds = []
            for batch_data in loader:
                pred_gene_values = model.pred_perturb(
                    batch_data, include_zero_gene, gene_ids=gene_ids, amp=amp
                )
                preds.append(pred_gene_values)
            preds = torch.cat(preds, dim=0)
            results_pred["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)

    return results_pred


def plot_perturbation(
    model: nn.Module, query: str, save_file: str = None, pool_size: int = None,
    pert_data = None,
):
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt

    sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)

    adata = pert_data.adata
    gene2idx = pert_data.node_map
    cond2name = dict(adata.obs[["condition", "condition_name"]].values)
    gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))

    de_idx = [
        gene2idx[gene_raw2id[i]]
        for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]
    ]
    genes = [
        gene_raw2id[i] for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]
    ]
    truth = adata[adata.obs.condition == query].X.toarray()[:, de_idx]
    if query.split("+")[1] == "ctrl":
        pred = predict(model, [[query.split("+")[0]]], pool_size=pool_size)
        pred = pred[query.split("+")[0]][de_idx]
    else:
        pred = predict(model, [query.split("+")], pool_size=pool_size)
        pred = pred["_".join(query.split("+"))][de_idx]
    ctrl_means = adata[adata.obs["condition"] == "ctrl"].to_df().mean()[de_idx].values

    pred = pred - ctrl_means
    truth = truth - ctrl_means

    plt.figure(figsize=[16.5, 4.5])
    plt.title(query)
    plt.boxplot(truth, showfliers=False, medianprops=dict(linewidth=0))

    for i in range(pred.shape[0]):
        _ = plt.scatter(i + 1, pred[i], color="red")

    plt.axhline(0, linestyle="dashed", color="green")

    ax = plt.gca()
    ax.xaxis.set_ticklabels(genes, rotation=90)

    plt.ylabel("Change in Gene Expression over Control", labelpad=10)
    plt.tick_params(axis="x", which="major", pad=5)
    plt.tick_params(axis="y", which="major", pad=5)
    sns.despine()

    if save_file:
        plt.savefig(save_file, bbox_inches="tight", transparent=False)

def pred_perturb_new(
    model,
    batch_data,
    include_zero_gene="batch-wise",
    gene_ids=None,
    amp=True,
    map_raw_id_to_vocab_id = None
):
    """
    Args:
        batch_data: a dictionary of input data with keys.

    Returns:
        output Tensor of shape [N, seq_len]
    """
    model.eval()
    device = next(model.parameters()).device
    batch_data.to(device)
    batch_size = len(batch_data.pert)
    x: torch.Tensor = batch_data.x
    # ori_gene_values = x[:, 0].view(batch_size, n_genes)
    ori_gene_values = x
    # pert_flags = x[:, 1].long().view(batch_size, n_genes)
    pert_flags = batch_data.pert_flags.long()

    if include_zero_gene in ["all", "batch-wise"]:
        assert gene_ids is not None
        if include_zero_gene == "all":
            input_gene_ids = torch.arange(ori_gene_values.size(1), device=device)
        else:  # batch-wise
            input_gene_ids = (
                ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
            )
        input_values = ori_gene_values[:, input_gene_ids]
        input_pert_flags = pert_flags[:, input_gene_ids]

        mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
        mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

        src_key_padding_mask = torch.zeros_like(
            input_values, dtype=torch.bool, device=device
        )
        with torch.cuda.amp.autocast(enabled=amp):
            output_dict = model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=False,
                CCE=False,
                MVC=False,
                ECS=False,
                do_sample=True,
            )
        output_values = output_dict["mlm_output"].float()
        pred_gene_values = torch.zeros_like(ori_gene_values)
        pred_gene_values[:, input_gene_ids] = output_values
    return pred_gene_values

def predict_new(
    model: TransformerGenerator, pert_list: List[str], pool_size: Optional[int] = None,
num_de_genes = 20, bs_test =32, shuffle_test=False,
    pert_data = None,
    include_zero_gene = None,
    gene_ids = None,
    amp = None
    ) -> Dict:
    """
    Predict the gene expression values for the given perturbations.

    Args:
        model (:class:`torch.nn.Module`): The model to use for prediction.
        pert_list (:obj:`List[str]`): The list of perturbations to predict.
        pool_size (:obj:`int`, optional): For each perturbation, use this number
            of cells in the control and predict their perturbation results. Report
            the stats of these predictions. If `None`, use all control cells.
    """
    adata = pert_data.adata
    # if pool_size is None:
    #     pool_size = len(ctrl_adata.obs)
    # - get total cell graphs for each dataset
    pert_cell_graphs = {}
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        results_pred = {}
        for pert in tqdm(pert_list):
            adata_pert = adata[adata.obs['perturbation_group']==pert]
            Xs = adata_pert.X # ctrl value
            ys = adata[adata_pert.obs['control_barcode']].X # perturb value
            
            # - get the de_idx for pert
            if 'rank_genes_groups' in adata_pert.uns:
                de_genes = adata_pert.uns['rank_genes_groups']
                de = True
            else:
                de = False
                # num_de_genes = 1
                
            if de:
                de_idx = np.where(adata_pert.var_names.isin(
                np.array(de_genes[pert][:num_de_genes])))[0]
            else:
                de_idx = [-1] * num_de_genes
                
            # - get the pert_idx [TODO]
            pert_idx = pert_data.get_pert_idx(pert)
            
            if not isinstance(Xs, np.ndarray):
                Xs = Xs.toarray()
            if not isinstance(ys, np.ndarray):
                ys = ys.toarray()

            pert_flags = torch.zeros(Xs.shape[1])
            if pert.split(' | ')[0] in adata.var_names:
                pert_flags[pert_data.var_idx_dict[pert.split(' | ')[0]]] = 1
            cell_graphs = []
            # Create cell graphs
            for X, y in zip(Xs, ys):
                # feature_mat = torch.Tensor(X).T
                if pert_idx is None:
                    pert_idx = [-1]
                gears_pert = '+'.join([pert.split(' | ')[0], 'ctrl']) # change to gears pert
                tmp_Data = Data(x=torch.Tensor(X.reshape(1,-1)), pert_idx=pert_idx,
                            y=torch.Tensor(y.reshape(1,-1)), de_idx=de_idx, pert=gears_pert,
                            pert_flags=pert_flags.reshape(1,-1))
                cell_graphs.append(tmp_Data)

            loader = DataLoader(cell_graphs,
                                        batch_size=bs_test, shuffle=shuffle_test)
            
            # - get model output
            preds = []
            for batch_data in loader:
                pred_gene_values = pred_perturb_new(model,
                    batch_data, include_zero_gene, gene_ids=gene_ids, amp=amp
                )
                preds.append(pred_gene_values)
            preds = torch.cat(preds, dim=0)
            results_pred[pert] = np.mean(preds.detach().cpu().numpy(), axis=0)

        return results_pred


    # adata = pert_data.adata
    # ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
    # if pool_size is None:
    #     pool_size = len(ctrl_adata.obs)
    # gene_list = pert_data.gene_names.values.tolist()
    # for pert in pert_list:
    #     for i in pert:
    #         if i not in gene_list:
    #             raise ValueError(
    #                 "The gene is not in the perturbation graph. Please select from GEARS.gene_list!"
    #             )

    # model.eval()
    # device = next(model.parameters()).device
    # with torch.no_grad():
    #     results_pred = {}
    #     for pert in pert_list:
    #         cell_graphs = create_cell_graph_dataset_for_prediction(
    #             pert, ctrl_adata, gene_list, device, num_samples=pool_size
    #         )
    #         loader = DataLoader(cell_graphs, batch_size=eval_batch_size, shuffle=False)
    #         preds = []
    #         for batch_data in loader:
    #             pred_gene_values = model.pred_perturb(
    #                 batch_data, include_zero_gene, gene_ids=gene_ids, amp=amp
    #             )
    #             preds.append(pred_gene_values)
    #         preds = torch.cat(preds, dim=0)
    #         results_pred["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)

    # return results_pred


def plot_perturbation_new(
    model: nn.Module, query: str, save_file: str = None, pool_size: int = None,
    pert_data = None,
    best_model = None,
    include_zero_gene = None,

):
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt

    sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)

    adata = pert_data.adata
    gene2idx = pert_data.node_map
    cond2name = dict(adata.obs[["condition", "condition_name"]].values)
    gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))

    de_idx = [
        gene2idx[gene_raw2id[i]]
        for i in adata.uns["top_non_dropout_de_20"][query]
    ]
    genes = [
        gene_raw2id[i] for i in adata.uns["top_non_dropout_de_20"][query]
    ]
    truth = adata[adata.obs['perturbation_group'] == query].X.toarray()[:, de_idx]

    pred = predict_new(best_model, [query])
    pred = pred[query][de_idx]

    # if query.split("+")[1] == "ctrl":
    #     pred = predict(model, [[query.split("+")[0]]], pool_size=pool_size)
    #     pred = pred[query.split("+")[0]][de_idx]
    # else:
    #     pred = predict(model, [query.split("+")], pool_size=pool_size)
    #     pred = pred["_".join(query.split("+"))][de_idx]

    ctrl_means = adata[adata.obs["condition"] == "ctrl"].to_df().mean()[de_idx].values

    pred = pred - ctrl_means
    truth = truth - ctrl_means

    plt.figure(figsize=[16.5, 4.5])
    plt.title(query)
    plt.boxplot(truth, showfliers=False, medianprops=dict(linewidth=0))

    for i in range(pred.shape[0]):
        _ = plt.scatter(i + 1, pred[i], color="red")

    plt.axhline(0, linestyle="dashed", color="green")

    ax = plt.gca()
    ax.xaxis.set_ticklabels(genes, rotation=90)

    plt.ylabel("Change in Gene Expression over Control", labelpad=10)
    plt.tick_params(axis="x", which="major", pad=5)
    plt.tick_params(axis="y", which="major", pad=5)
    sns.despine()

    if save_file:
        plt.savefig(save_file, bbox_inches="tight", transparent=False)

def eval_perturb_new(
    loader: DataLoader, model: TransformerGenerator, device: torch.device,
    include_zero_gene,
    gene_ids
) -> Dict:
    """
    Run model in inference mode using a given data loader
    """

    model.eval()
    model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}
    logvar = []

    for itr, batch in enumerate(loader):
        batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.no_grad():
            p = pred_perturb_new(model, batch, include_zero_gene, gene_ids=gene_ids)
            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())

            # Differentially expressed genes
            for itr, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[itr, de_idx])
                truth_de.append(t[itr, de_idx])

    # all genes
    results["pert_cat"] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results["pred"] = pred.detach().cpu().numpy().astype(np.float64)
    results["truth"] = truth.detach().cpu().numpy().astype(np.float64)

    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results["pred_de"] = pred_de.detach().cpu().numpy().astype(np.float64)
    results["truth_de"] = truth_de.detach().cpu().numpy().astype(np.float64)

    return results