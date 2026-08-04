import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader
from MPG.Models import GCN, Prompt
from prompt_w_o_h import eva, multi_class_data
from utils import seed_everything, seed
seed_everything(seed)


def model_create(dataname, gnn_type, num_class, task_type='multi_class_classification'):

    input_dim, hid_dim = 100, 100
    pre_method = 'SimGRACE'
    token_num = 10

    gnn = GCN(input_dim, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=2, gnn_type=gnn_type)
    pre_train_path = './pre_trained_gnn/{}.{}.{}.pth'.format(dataname, pre_method, gnn_type)
    gnn.load_state_dict(torch.load(pre_train_path))
    print("successfully load pre-trained weights for gnn! @ {}".format(pre_train_path))
    for p in gnn.parameters():
        p.requires_grad = False

    if task_type == 'regression':
        task_head = torch.nn.Sequential(
            torch.nn.Linear(hid_dim, num_class),
            torch.nn.Sigmoid())
    else:
        task_head = torch.nn.Sequential(
            torch.nn.Linear(hid_dim, num_class),
            torch.nn.Softmax(dim=1))

    prompt = Prompt(token_dim=input_dim, token_num=token_num, prune_thre=0.1,
                    isolate_tokens=False, inner_prune=0.9)

    opi_head = optim.Adam(filter(lambda p: p.requires_grad, task_head.parameters()), lr=0.01,
                          weight_decay=0.00001)

    opi_prompt = optim.Adam(filter(lambda p: p.requires_grad, prompt.parameters()), lr=0.01,
                            weight_decay=0.00001)

    if task_type == 'regression':
        lossfn = nn.MSELoss(reduction='mean')
    else:
        lossfn = nn.CrossEntropyLoss(reduction='mean')

    return gnn, task_head, prompt, opi_head, opi_prompt, lossfn

def testing(test, prompt, gnn, task_head, task_type='multi_class_classification'):
    task_head.eval()
    prompt.eval()

    if task_type != 'link_prediction':
        xp, xp_edge_index, batch_one, _ = prompt(test)
        graph_emb = gnn(xp, xp_edge_index, batch_one)
        pre = task_head(graph_emb)
        pre = pre.detach()
        res = eva(pre, test.y, task_type=task_type)
        return res
    else:
        data_list = test.to_data_list()
        batch_size = 10
        loader = DataLoader(data_list,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=1)

        pre_list = []
        label_list = []
        for step, data in enumerate(loader):
            xp, xp_edge_index, batch_one, _ = prompt(data)
            graph_emb = gnn(xp, xp_edge_index, batch_one)
            pre = task_head(graph_emb)
            pre = pre.detach()
            pre_list.append(pre)
            label_list.append(data.y)

        pre = torch.cat(pre_list, dim=0).numpy()
        y = torch.cat(label_list).numpy()
        res = eva(pre, y, task_type=task_type)
        return res


if __name__ == '__main__':

    dataname, gnn_type, num_class = "CiteSeer", "TransformerConv", 6

    train, test = multi_class_data(dataname, num_class)
    task_type = 'multi_class_classification'

    gnn, task_head, prompt, opi_head, opi_prompt, lossfn = model_create(dataname, gnn_type, num_class, task_type)

    outer_epoch = 10
    # inspired by: Hou Y et al. MetaPrompting: Learning to Learn Better Prompts. COLING 2022
    # if we use the task_head, we update task_head and prompt alternately.
    # ignore the outer_epoch if you do not use any tunable task_head
    head_epoch = 50
    prompt_epoch = 50

    # training stage
    for i in range(1, outer_epoch + 1):
        print("{}/{} frozen gnn | frozen prompt | *tune task head...".format(i, outer_epoch))
        # tune task head
        task_head.train()
        prompt.eval()
        for j in range(1, head_epoch + 1):
            xp, xp_edge_index, batch_one, _ = prompt(train)
            graph_emb = gnn(xp, xp_edge_index, batch_one)
            pre = task_head(graph_emb)
            train_loss = lossfn(pre, train.y)
            if j % 5 == 0:
                print('\t\t==> {}/{} training loss: {:.8f}'.format(j, head_epoch, train_loss.item()))
            opi_head.zero_grad()
            train_loss.backward()
            opi_head.step()

        # testing stage
        res = testing(test, prompt, gnn, task_head, task_type)
        if task_type == 'regression':
            print("""Fine-Tune @ {}/{} | MAE: {:.4} | MSE: {:.4} """.format(i, outer_epoch, res["mae"], res["mse"]))
        elif task_type == 'link_prediction':
            print("""Fine-Tune @ {}/{} | MRR, Hits:{} """.format(i, outer_epoch, res))
        else:
            print("""Fine-Tune @ {}/{} | Acc: {:.4} | Macro F1: {:.4} | Micro F1: {:.4}""".format(i, outer_epoch,
                                                                                                  res["acc"],
                                                                                                  res["mac_f1"],
                                                                                                  res["mic_f1"]))

        print("{}/{}  frozen gnn | *tune prompt |frozen task head...".format(i, outer_epoch))
        # tune prompt
        task_head.eval()
        prompt.train()
        for k in range(1, prompt_epoch + 1):
            xp, xp_edge_index, batch_one, _ = prompt(train)
            graph_emb = gnn(xp, xp_edge_index, batch_one)
            pre = task_head(graph_emb)
            train_loss = lossfn(pre, train.y)
            if k % 5 == 0:
                print('\t\t==> {}/{} training loss: {:.8f}'.format(k, prompt_epoch, train_loss.item()))
            opi_prompt.zero_grad()
            train_loss.backward()
            opi_prompt.step()

        # testing stage
        res = testing(test, prompt, gnn, task_head, task_type)
        if task_type == 'regression':
            print("""Prompt @ {}/{} | MAE: {:.4} | MSE: {:.4} """.format(i, outer_epoch, res["mae"], res["mse"]))
        elif task_type == 'link_prediction':
            print(
                """Prompt @ {}/{} | MRR, Hits:{} """.format(i, outer_epoch, res))
        else:
            print("""Prompt @ {}/{} | Acc: {:.4} | Macro F1: {:.4} | Micro F1: {:.4}""".format(i, outer_epoch,
                                                                                               res["acc"],
                                                                                               res["mac_f1"],
                                                                                               res["mic_f1"]))

    # testing stage
    res = testing(test, prompt, gnn, task_head, task_type)
    if task_type == 'regression':
        print("""FINAL| MAE: {:.4} | MSE: {:.4} """.format(res["mae"], res["mse"]))
    elif task_type == 'link_prediction':
        print("""FINAL| | MRR, Hits:{} """.format(i, outer_epoch, res))
    else:
        print("""FINAL | Acc: {:.4} | Macro F1: {:.4} | Micro F1: {:.4}""".format(res["acc"],
                                                                                  res["mac_f1"],
                                                                                  res["mic_f1"]))
