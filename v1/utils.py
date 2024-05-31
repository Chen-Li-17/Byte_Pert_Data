import scanpy as sc
import pandas as pd
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import sys
import requests

def print_sys(s):
    """system print

    Args:
        s (str): the string to print
    """
    print(s, flush = True, file = sys.stderr)

def dataverse_download(url, save_path):
    """
    Dataverse download helper with progress bar

    Args:
        url (str): the url of the dataset
        path (str): the path to save the dataset
    """
    
    if os.path.exists(save_path):
        print_sys('Found local copy...')
    else:
        print_sys("Downloading...")
        response = requests.get(url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        
def get_genes_from_perts(perts):
    """
    Returns list of genes involved in a given perturbation list
    """

    if type(perts) is str:
        perts = [perts]
    gene_list = [p.split('+') for p in np.unique(perts)]
    gene_list = [item for sublist in gene_list for item in sublist]
    gene_list = [g for g in gene_list if g != 'ctrl']
    return list(np.unique(gene_list))

def get_dropout_non_zero_genes(adata):
    
    # calculate mean expression for each condition
    unique_conditions = adata.obs.condition.unique()
    conditions2index = {}
    for i in unique_conditions:
        conditions2index[i] = np.where(adata.obs.condition == i)[0]

    condition2mean_expression = {}
    for i, j in conditions2index.items():
        condition2mean_expression[i] = np.mean(adata.X[j], axis = 0)
    pert_list = np.array(list(condition2mean_expression.keys()))
    mean_expression = np.array(list(condition2mean_expression.values())).reshape(len(adata.obs.condition.unique()), adata.X.shape[1])
    ctrl = mean_expression[np.where(pert_list == 'ctrl')[0]]
    
    ## in silico modeling and upperbounding
    pert2pert_full_id = dict(adata.obs[['condition', 'condition_name']].values)
    pert_full_id2pert = dict(adata.obs[['condition_name', 'condition']].values)

    gene_id2idx = dict(zip(adata.var.index.values, range(len(adata.var))))
    gene_idx2id = dict(zip(range(len(adata.var)), adata.var.index.values))

    non_zeros_gene_idx = {}
    top_non_dropout_de_20 = {}
    top_non_zero_de_20 = {}
    non_dropout_gene_idx = {}

    for pert in adata.uns['rank_genes_groups'].keys():
        p = pert_full_id2pert[pert]
        # X = np.mean(adata[adata.obs.condition == p].X, axis = 0)
        X = np.mean(adata[adata.obs.condition_name == pert].X, axis = 0)

        non_zero = np.where(np.array(X)[0] != 0)[0]
        zero = np.where(np.array(X)[0] == 0)[0]
        true_zeros = np.intersect1d(zero, np.where(np.array(ctrl)[0] == 0)[0])
        non_dropouts = np.concatenate((non_zero, true_zeros))

        top = adata.uns['rank_genes_groups'][pert]
        gene_idx_top = [gene_id2idx[i] for i in top]

        non_dropout_20 = [i for i in gene_idx_top if i in non_dropouts][:20]
        non_dropout_20_gene_id = [gene_idx2id[i] for i in non_dropout_20]

        non_zero_20 = [i for i in gene_idx_top if i in non_zero][:20]
        non_zero_20_gene_id = [gene_idx2id[i] for i in non_zero_20]

        non_zeros_gene_idx[pert] = np.sort(non_zero)
        non_dropout_gene_idx[pert] = np.sort(non_dropouts)
        top_non_dropout_de_20[pert] = np.array(non_dropout_20_gene_id)
        top_non_zero_de_20[pert] = np.array(non_zero_20_gene_id)
        
    non_zero = np.where(np.array(X)[0] != 0)[0]
    zero = np.where(np.array(X)[0] == 0)[0]
    true_zeros = np.intersect1d(zero, np.where(np.array(ctrl)[0] == 0)[0])
    non_dropouts = np.concatenate((non_zero, true_zeros))
    
    adata.uns['top_non_dropout_de_20'] = top_non_dropout_de_20
    adata.uns['non_dropout_gene_idx'] = non_dropout_gene_idx
    adata.uns['non_zeros_gene_idx'] = non_zeros_gene_idx
    adata.uns['top_non_zero_de_20'] = top_non_zero_de_20
    
    return adata

def get_pert_celltype(pert_group_list):
    perts = np.unique([i.split(' | ')[0] for i in pert_group_list])
    celltypes = np.unique([i.split(' | ')[1] for i in pert_group_list])
    return perts, celltypes


def plot_loss(tmp_list,
              # test_list,
              dataset,
              name):
    
    import matplotlib.pyplot as plt
    # 假设损失函数值存储在 loss_values 中
    # plt.figure(figsize=(8,6))

    # 创建 x 轴的值（可以是迭代次数、时间步长等）
    iterations = range(1, len(tmp_list) + 1)

    # 绘制损失函数曲线
    plt.plot(iterations, tmp_list, 
            #  marker='o', 
             linestyle='-', 
             label='train')
    # plt.plot(iterations, test_list, marker='o', linestyle='-', label='test')
    

    # 添加标题和标签
    plt.title(f'{name} Curve - {dataset}')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    # plt.legend()

    # 显示网格
    plt.grid(True)

    # # 显示图形
    # plt.show()
    
def merge_plot(metrics_list, dataset, save_dir=None):
    
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    tmp_list = [i['mse'] for i in metrics_list]
    plot_loss(tmp_list, dataset, 'MSE')

    plt.subplot(2, 2, 2)
    tmp_list = [i['pearson'] for i in metrics_list]
    plot_loss(tmp_list, dataset, 'PCC')

    plt.subplot(2, 2, 3)
    tmp_list = [i['mse_de'] for i in metrics_list]
    plot_loss(tmp_list, dataset, 'MSE_de')

    plt.subplot(2, 2, 4)
    tmp_list = [i['pearson_de'] for i in metrics_list]
    plot_loss(tmp_list, dataset, 'PCC_de')

    # 调整子图之间的间距
    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir)
        
def get_info_txt(pert_data = None, 
                 save_dir = None):

    # - get the para setting
    basic_para = f'\
    pert_cell_filter: {pert_data.pert_cell_filter} # this is a test \n\
    seed:             {pert_data.seed} # this is the random seed\n\
    split_type:       {pert_data.split_type} # 1 for unseen perts; 0 for unseen celltypes \n\
    split_ratio:      {pert_data.split_ratio} # train:test:val; val is used to choose data, test is for final validation \n\
    var_num:          {pert_data.var_num} # selecting hvg number \n\
    num_de_genes:     {pert_data.num_de_genes} # number of de genes \n\
    bs_train:         {pert_data.bs_train} # batch size of trainloader \n\
    bs_test:          {pert_data.bs_test} # batch size of testloader \n\
    '

    # - get the pert info
    ori_pert_num = pert_data.value_counts.shape[0]
    filtered_filter_num = pert_data.value_counts[pert_data.value_counts['Counts']<pert_data.pert_cell_filter].shape[0]
    final_pert_num = ori_pert_num - filtered_filter_num
    pert_num = len(pert_data.filter_perturbation_list)
    pert_info = f'\
    Number of original perts is: {ori_pert_num}\n\
    Number of filtered perts is: {filtered_filter_num}\n\
    After filter, number of non-ctrl perts: {pert_num}\n\
    After filter, number of pert genes: {len(pert_data.pert_gene_list)}\n\
    '
    # print(pert_info)


    # - get the gene_info
    exclude_var_num = len(pert_data.exclude_var_list)
    exclude_pert_num = 0
    for pert in pert_data.filter_perturbation_list:
        for gene in pert.split(' | ')[0].split('; '):
            if gene not in pert_data.adata_split.var_names:
                exclude_pert_num += 1
                break

    exclude_gene_num_gears = 0
    for gene in pert_data.pert_gene_list:
        if gene not in pert_data.pert_names:
            exclude_gene_num_gears += 1
    
    exclude_pert_num_gears = 0
    for pert in pert_data.filter_perturbation_list:
        for gene in pert.split(' | ')[0].split('; '):
            if gene not in pert_data.pert_names:
                exclude_pert_num_gears += 1
                break
            
    gene_info = f'\
    {exclude_var_num}/{len(pert_data.pert_gene_list)} genes are not in var names\n\
    {exclude_pert_num}/{len(pert_data.filter_perturbation_list)} perts are not in var names\n\
    \n\
    {exclude_gene_num_gears}/{len(pert_data.pert_gene_list)} genes are not in GEARS.pert_names\n\
    {exclude_pert_num_gears}/{len(pert_data.filter_perturbation_list)} perts are not in GEARS.pert_names\n\
    '
    # print(gene_info)

    # - get the adata_info
    pert_count_dict = {}
    for pert in pert_data.filter_perturbation_list:
        pert_count_dict[pert] = len(pert_data.adata_split[pert_data.adata_split.obs['perturbation_group']==pert])
    ctrl_count = len(pert_data.adata_split[pert_data.adata_split.obs['perturbation_new']=='control'])
    adata_info = f'\
    adata shape: {(pert_data.adata_split.shape)}\n\
    pert average cell num: {int(np.mean(list(pert_count_dict.values())))}\n\
    ctrl cell num: {int(ctrl_count)}\n\
    '
    # print(adata_info)

    from tabulate import tabulate

    # - get the sgRNA info
    sgRNA_info = f'\
    {len(pert_data.multi_sgRNA_pert_list)}/{len(pert_data.filter_perturbation_list)} perts have more than 1 sgRNA\n\
    {len(pert_data.df_sgRNA_edis_dict)}/{len(pert_data.multi_sgRNA_pert_list)} perts are in the var_names\n\
    {len(pert_data.filter_pert_sgRNA)} pert_sgRNA are filtered\n\
    '
    # print(sgRNA_info)

    # - get the split info
    split_info = f'\
    perts num of train: val: test = {len(pert_data.train_perts)}: {len(pert_data.val_perts)}: {len(pert_data.test_perts)} \n\
    '
    # print(split_info)

    # 指定文件名
    filename = os.path.join(save_dir, f"INFO_{pert_data.prefix}.txt")

    # 将信息写入文件
    with open(filename, "w") as file:
        file.write('*'*20+'Parameter Setting'+'*'*20+'\n')
        file.write(basic_para)
        file.write('\n')
        
        file.write('*'*20+'Pert Info'+'*'*20+'\n')
        file.write(pert_info)
        file.write('\n')
        
        file.write('*'*20+'Gene Info'+'*'*20+'\n')
        file.write(gene_info)
        file.write('\n')
        
        file.write('*'*20+'adata Info'+'*'*20+'\n')
        file.write(adata_info)
        file.write('\n')
        
        file.write('*'*20+'Data Split Info'+'*'*20+'\n')
        file.write(split_info)
        file.write('\n')
        
        file.write('*'*20+'Top 10 pert of edis'+'*'*20+'\n')
        file.write(tabulate(pert_data.df_pert_edis[0:10].round(2), headers='keys', tablefmt='mixed_grid'))
        file.write('\n')
        
        file.write('*'*20+'sgRNA Info'+'*'*20+'\n')
        file.write(sgRNA_info)
        file.write('\n')