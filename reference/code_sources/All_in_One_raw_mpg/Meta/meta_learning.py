import MPG
from torch import nn, optim
from data_load import load_tasks_Reddit2
from MPG.Models import Pipeline
import torch
from copy import deepcopy
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from utils import seed_everything, seed
from random import shuffle
import pandas as pd

seed_everything(seed)


def meta_test_adam(meta_test_task_id_list,
                   dataname,
                   K_shot,
                   seed,
                   maml,
                   adapt_steps_meta_test,
                   lossfn,
                   save_project_head=False,
                   save_pickles=None):
    pre_train_method, with_prompt, meta_learning, gnn_type, meta_test_type = save_pickles

    task_results = []
    # meta-testing
    if len(meta_test_task_id_list) < 2:
        raise AttributeError("\ttask_id_list should contain at leat two tasks!")

    shuffle(meta_test_task_id_list)

    task_pairs = [(meta_test_task_id_list[i], meta_test_task_id_list[i + 1]) for i in
                  range(0, len(meta_test_task_id_list) - 1, 2)]

    for task_1, task_2, support, query, _ in load_tasks_Reddit2('test',
                                                                task_pairs,
                                                                dataname,
                                                                K_shot,
                                                                seed):

        test_model = deepcopy(maml.module)
        test_opi = optim.Adam(filter(lambda p: p.requires_grad, test_model.parameters()),
                              lr=0.001,
                              weight_decay=0.00001)

        test_model.train()

        for _ in range(adapt_steps_meta_test):
            support_preds = test_model(support)
            support_loss = lossfn(support_preds, support.y)
            if _ % 5 == 0:
                print('{}/{} training loss: {:.8f}'.format(_,
                                                           adapt_steps_meta_test,
                                                           support_loss.item()))
            test_opi.zero_grad()
            support_loss.backward()
            test_opi.step()

        test_model.eval()
        query_preds = test_model(query)

        pre_class = torch.argmax(query_preds, dim=1)
        acc = accuracy_score(query.y, pre_class)
        f1 = f1_score(query.y, pre_class, average='binary')
        auc = roc_auc_score(query.y, query_preds[:, 1].detach().numpy())
        print("""\ttask pair ({}, {}) | Acc: {:.4} | F1: {:.4} | ACU: {:.4}""".format(task_1, task_2, acc, f1, auc))
        task_results.append([task_1, task_2, acc, f1, auc])

        if save_project_head:
            torch.save(test_model.project_head.state_dict(),
                       "./project_head/{}.{}.{}.{}.pth".format(dataname, pre_train_method, gnn_type, meta_test_type))
            print("project head saved! @./project_head/{}.{}.{}.{}.pth".format(dataname, pre_train_method, gnn_type,
                                                                               meta_test_type))

    return task_results


def meta_train_maml(epoch, maml, lossfn, opt, meta_train_task_id_list, dataname, adapt_steps, K_shot=100):
    if len(meta_train_task_id_list) < 2:
        raise AttributeError("\ttask_id_list should contain at leat two tasks!")

    shuffle(meta_train_task_id_list)

    task_pairs = [(meta_train_task_id_list[i], meta_train_task_id_list[i + 1]) for i in
                  range(0, len(meta_train_task_id_list) - 1, 2)]

    # meta-training
    for ep in range(epoch):
        meta_train_loss = 0.0
        pair_count = 0

        for task_1, task_2, support, query, total_num in load_tasks_Reddit2('train',
                                                                            task_pairs,
                                                                            dataname,
                                                                            K_shot,
                                                                            seed):
            pair_count = pair_count + 1

            learner = maml.clone()

            for _ in range(adapt_steps):  # adaptation_steps
                support_preds = learner(support)
                support_loss = lossfn(support_preds, support.y)
                learner.adapt(support_loss)

            query_preds = learner(query)
            query_loss = lossfn(query_preds, query.y)
            meta_train_loss += query_loss

        print('\tmeta_train_loss at epoch {}/{}: {}'.format(ep, epoch, meta_train_loss.item()))
        meta_train_loss = meta_train_loss / len(meta_train_task_id_list)
        opt.zero_grad()
        meta_train_loss.backward()
        opt.step()


def model_components(args, round=1, pre_train_path='', gnn_type='', project_head_path=None):
    if round == 1:
        model = Pipeline(input_dim=602,
                         pre_train_path=pre_train_path, gcn_layer_num=2, hid_dim=100, num_classes=2,
                         frozen_gnn='all',
                         with_prompt=False, token_num=5, prune_thre=0.9, inner_prune=0.9,
                         isolate_tokens=False,
                         frozen_project_head=False, pool_mode=1, gnn_type=gnn_type, project_head_path=None)
    elif round == 2 or round == 3:
        if project_head_path is None:
            raise ValueError("project_head_path is None! it should be a specific path when round=2 or 3")
        model = Pipeline(input_dim=602,
                         pre_train_path=pre_train_path, gcn_layer_num=2, hid_dim=100, num_classes=2,
                         frozen_gnn='all',
                         with_prompt=True, token_num=5, prune_thre=0.9, inner_prune=0.9,
                         isolate_tokens=False,
                         frozen_project_head=True, pool_mode=1, gnn_type=gnn_type, project_head_path=project_head_path)
    else:
        raise ValueError('round value wrong! (it should be 1,2,3)')

    maml = MPG.MAML(model, lr=args.adapt_lr, first_order=False, allow_nograd=True)
    opt = optim.Adam(filter(lambda p: p.requires_grad, maml.parameters()), args.meta_lr)
    lossfn = nn.CrossEntropyLoss(reduction='mean')

    return maml, opt, lossfn


if __name__ == '__main__':

    from config import ParReddit2

    args = ParReddit2()

    res_full = []
    for source_level, task_arrange in args.exp_type.items():
        meta_train_task_id_list = task_arrange["meta_train_tasks"]

        res_per_source = []

        for paras in args.para_set:

            pre_train_method, with_prompt, meta_learning, gnn_type, pre_train_path = paras

            maml, opt, lossfn = model_components(args)

            if meta_learning:
                # meta-training
                print("meta-training for {}.{}.{}.{}.{}...".format(source_level, pre_train_method, with_prompt,
                                                                   meta_learning, gnn_type))
                meta_train_maml(args.epoch, maml, lossfn, opt, meta_train_task_id_list,
                                args.dataname, args.adapt_steps, K_shot=args.K_shot)

            for meta_test_type, meta_test_task_id_list in task_arrange["meta_test_tasks"].items():
                print("meta-test for {}.{}.{}.{}.{}.{}...".format(source_level, pre_train_method, with_prompt,
                                                                  meta_learning, gnn_type, meta_test_type))
                save_pickles = (
                    source_level, pre_train_method, with_prompt, meta_learning, gnn_type, meta_test_type)
                res = meta_test_adam(meta_test_task_id_list, args.dataname, args.K_shot, seed, maml,
                                     args.adapt_steps_meta_test, lossfn, save_project_head=False,
                                     save_pickles=save_pickles)

                res_per_source.append([source_level, meta_test_type, pre_train_method,
                                       with_prompt, meta_learning, gnn_type] + res)

        res_per_source_pd = pd.DataFrame(res_per_source)

        res_per_source_pd.columns = ['source', 'target', 'PTM', 'prompt', 'meta',
                                     'gnn_type', 'task_1_id', 'task_2_id', 'acc', 'f1', 'auc']

        path_per_source = './results/{}.{}.xlsx'.format(args.dataname, source_level)

        res_per_source_pd.to_excel(path_per_source, header=True, index=False)

    res_full_pd = pd.DataFrame(res_full)

    res_full_pd.columns = ['source', 'target', 'PTM', 'prompt', 'meta',
                           'gnn_type', 'task_1_id', 'task_2_id', 'acc', 'f1', 'auc']
    path_res_full = './results/{}.full.xlsx'.format(args.dataname)

    res_full_pd.to_excel(path_res_full, header=True, index=False)

    print("ALL DONE!")
