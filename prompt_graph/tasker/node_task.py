import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data   import Data
import torch.utils.data as Data1
from prompt_graph.utils import constraint,  center_embedding, Gprompt_tuning_loss, process, cmd, MLP, train_MLP, get_psu_labels, finetune_answering, get_detector
from prompt_graph.evaluation import GPPTEva, GNNNodeEva, GpromptEva, MultiGpromptEva, GPFEva, AllInOneEva, RobustPromptInductiveEva, RobustPromptTranductiveEva, GPFTranductiveEva
from prompt_graph.data import induced_graphs, split_induced_graphs, split_induced_graphs_save_relabel_central_node_and_raw_index, load4cora_downstream_clean, load4node_attack_shot_index, load4node_attack_specified_shot_index
from  easydict  import EasyDict

from .task import BaseTask
import time
import warnings

import pickle
import os
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.data import Batch, Data
from tqdm import tqdm
import copy 
from project_paths import attack_data_root, project_path
from prompt_graph.utils.edge_anomaly_metrics import (
      evaluate_edge_detection,
      evaluate_incident_edge_detection,
      evaluate_node_detection,
      fmt_metric,
      pollution_diff,
)
from types import SimpleNamespace

warnings.filterwarnings("ignore")

class NodeTask(BaseTask):
      def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.task_type = 'NodeTask'
            self.edge_detection_reference = None

            if self.attack_downstream:
                  # assert self.attack_method != 'None', 'No specific attacks were designated.'
                  self.load_shot_attack_data()
            else:
                  self.load_data()

            self.initialize_gnn()
            self.initialize_prompt()
            self.answering =  torch.nn.Linear(self.hid_dim, self.output_dim).to(self.device)
            self.initialize_optimizer()
      


      def process_multigprompt_data(self, data):
            data = data.cpu()
            adj = to_scipy_sparse_matrix(data.edge_index).tocsr()
            # Convert features to dense format and then to scipy sparse matrix in lil format
            features = sp.lil_matrix(data.x.numpy())
            # Convert labels to one-hot encoding
            labels = np.zeros((data.num_nodes, data.y.max().item() + 1))
            labels[np.arange(data.num_nodes), data.y.numpy()] = 1

            # adj, features, labels = process.load_data(self.dataset_name)
            # adj, features, labels = process.load_data(self.dataset_name)  
            self.input_dim = features.shape[1]
            self.output_dim = labels.shape[1]

            features, _ = process.preprocess_features(features)
            self.sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj).to(self.device)
            self.labels = torch.FloatTensor(labels[np.newaxis])
            self.features = torch.FloatTensor(features[np.newaxis]).to(self.device)
            # print("labels",labels)
            print("adj",self.sp_adj.shape)
            print("feature",features.shape)



      def load_shot_attack_data(self):
            if self.specified:
                  # 对指定的train/val/test划分方式进行攻击，因为一些方法对不同的划分会产生不同的分布，这样能更加精准的实施攻击，但是对于每一个攻击都要实施一下，攻击成本更大
                  print("load LLC or attacked data with specified split")
                  data_dir_name = 'data_attack_fewshot' 
                  self.data, self.dataset = load4node_attack_specified_shot_index(data_dir_name, self.dataset_name, self.attack_method, shot_num = self.shot_num, run_split= self.run_split)
            else:
                  # 已经存在对默认的train/val/test划分方式进行攻击的数据集，从默认数据集中的train中提取不同shot的index, 这样更科学，从默认的攻击划分中进行抽取而不是从全局随机抽取可以相对保留攻击的效果
                  print('load LLC or attacked shot data with default split')
                  if self.adaptive:
                        data_dir_name = 'data_unit_attack_from_default_split' 
                        adaptive_dict = EasyDict()
                        adaptive_dict['PARAM'] = {
                              "scenario": self.adaptive_scenario, 
                              "split":    self.adaptive_split, 
                              "adaptive_attack_model": self.adaptive_attack_model, 
                              "ptb_rate":  self.adaptive_ptb_rate
                        }
                        self.data, self.dataset = load4node_attack_shot_index(data_dir_name, self.dataset_name, self.attack_method, shot_num = self.shot_num, run_split= self.run_split, adaptive=self.adaptive, adaptive_dict = adaptive_dict['PARAM'])
                  else:
                        data_dir_name = 'data_attack_from_default_split' 
                        self.data, self.dataset = load4node_attack_shot_index(data_dir_name, self.dataset_name, self.attack_method, shot_num = self.shot_num, run_split= self.run_split)


            if self.prompt_type == 'MultiGprompt':
                  self.process_multigprompt_data(self.data)
            else:
                  self.input_dim = self.data.x.shape[1]
                  self.output_dim = self.dataset.num_classes

            self._prepare_edge_detection_reference(data_dir_name)




            if self.prompt_type in ['All-in-one','Gprompt', 'GPF', 'GPF-plus','RobustPrompt-I', 'RobustPrompt-I-original']:
                  if self.adaptive:
                        file_dir = './{}/{}/{}/{}/{}/{}/shot_{}/{}/induced_graph/'.format(data_dir_name, self.dataset_name, self.adaptive_scenario, str(self.adaptive_split), self.adaptive_attack_model, str(self.adaptive_ptb_rate), str(self.shot_num), str(self.run_split))
                  else:
                        file_dir = './{}/{}/shot_{}/{}/induced_graph/{}'.format(data_dir_name, self.dataset_name, str(self.shot_num), str(self.run_split), self.attack_method)
                  file_path = os.path.join(file_dir, 'induced_graph.pkl')
                  # 注意，换shot num的时候要把induced graph删掉
                  if os.path.exists(file_path):
                        # print('Begin load induced_graphs with specified shot {} and run split {} under {}.'.format(str(self.shot_num), str(self.run_split), self.attack_method))
                        with open(file_path, 'rb') as f:
                              graphs_dict = pickle.load(f)
                        self.train_dataset = graphs_dict['train_graphs']
                        self.test_dataset = graphs_dict['test_graphs']
                        self.val_dataset = graphs_dict['val_graphs']
                  else:
                        os.makedirs(file_dir, exist_ok=True) 
                        # print('Begin split induced_graphs with specified shot {} and run split {} under {}.'.format(str(self.shot_num), str(self.run_split), self.attack_method))
                        # split_induced_graphs(self.dataset_name, self.data, file_path, smallest_size=100, largest_size=300)

                        # smallest_size = 5  # 默认为5
                        # if self.dataset_name in ['ENZYMES', 'PROTEINS']:
                        #       smallest_size = 1
                        # if self.dataset_name == 'PubMed':
                        #       smallest_size = 8

                        split_induced_graphs_save_relabel_central_node_and_raw_index(self.dataset_name, self.data, file_path, smallest_size=100, largest_size=300)
                        with open(file_path, 'rb') as f:
                              graphs_dict = pickle.load(f)
                        self.train_dataset = graphs_dict['train_graphs']
                        self.test_dataset = graphs_dict['test_graphs']
                        self.val_dataset = graphs_dict['val_graphs']
            else:
                  self.data.to(self.device)



      def load_data(self):
            self.data, self.dataset = load4cora_downstream_clean(self.dataset_name, shot_num = self.shot_num, run_split= self.run_split)

            if self.prompt_type == 'MultiGprompt':
                  self.process_multigprompt_data(self.data)
            else:
                  self.input_dim = self.data.x.shape[1]
                  self.output_dim = self.dataset.num_classes


            if self.prompt_type in ['All-in-one','Gprompt', 'GPF', 'GPF-plus','RobustPrompt-I', 'RobustPrompt-I-original']:
                  # file_dir = './data/{}/induced_graph/shot_{}/{}'.format(self.dataset_name, str(self.shot_num), str(self.run_split))
                  # file_path = os.path.join(file_dir, 'induced_graph.pkl')

                  file_dir = './data_fewshot/{}/shot_{}/{}/induced_graph/'.format(self.dataset_name, str(self.shot_num), str(self.run_split))
                  file_path = os.path.join(file_dir, 'induced_graph.pkl')


                  # 注意，换shot num的时候要把induced graph删掉
                  if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                              graphs_dict = pickle.load(f)
                        self.train_dataset = graphs_dict['train_graphs']
                        self.test_dataset = graphs_dict['test_graphs']
                        self.val_dataset = graphs_dict['val_graphs']
                  else:
                        os.makedirs(file_dir, exist_ok=True) 
                        print('Begin split_induced_graphs.')
                        split_induced_graphs(self.dataset_name, self.data, file_path, smallest_size=5, largest_size=300)
                        with open(file_path, 'rb') as f:
                              graphs_dict = pickle.load(f)
                        self.train_dataset = graphs_dict['train_graphs']
                        self.test_dataset = graphs_dict['test_graphs']
                        self.val_dataset = graphs_dict['val_graphs']
            else:
                  self.data.to(self.device)
            

      def _load_clean_reference_data(self, data_dir_name):
            if self.adaptive or not self.attack_method or '-' not in self.attack_method:
                  return None

            atk_type = self.attack_method.split('-')[0]
            if self.specified:
                  from data_attack_fewshot.attackdata_specified import AttackDataset_specified

                  path = project_path(
                        data_dir_name,
                        self.dataset_name,
                        'shot_{}'.format(self.shot_num),
                        str(self.run_split),
                  )
                  dataset = AttackDataset_specified(
                        root=path,
                        name='Attack-' + self.dataset_name,
                        attackmethod=atk_type,
                        ptb_rate='0.0',
                  )
            else:
                  from data_pyg.data_pyg import get_dataset

                  dataset = get_dataset(
                        attack_data_root(),
                        'Attack-' + self.dataset_name,
                        atk_type,
                        0.0,
                  )
            return dataset[0]


      def _prepare_edge_detection_reference(self, data_dir_name):
            clean_data = self._load_clean_reference_data(data_dir_name)
            if clean_data is None:
                  self.edge_detection_reference = None
                  return

            if clean_data.num_nodes != self.data.num_nodes:
                  print(
                        "Pollution Diff | skipped=true | reason=num_nodes_mismatch "
                        f"| clean_nodes={clean_data.num_nodes} | attack_nodes={self.data.num_nodes}"
                  )
                  self.edge_detection_reference = None
                  return

            diff = pollution_diff(clean_data.edge_index, self.data.edge_index, undirected=True)
            self.edge_detection_reference = {
                  "clean_data": clean_data,
                  "clean_edges": diff["clean_edges"],
                  "attack_edges": diff["attack_edges"],
                  "added_edges": diff["added_edges"],
                  "deleted_edges": diff["deleted_edges"],
            }
            print(
                  "Pollution Diff | "
                  f"added_edges={len(diff['added_edges'])} | "
                  f"deleted_edges={len(diff['deleted_edges'])} | "
                  f"clean_edges={len(diff['clean_edges'])} | "
                  f"attack_edges={len(diff['attack_edges'])}"
            )


      def _print_edge_detection_metrics(self, source, metrics, extra=None):
            if metrics is None:
                  return
            parts = [
                  "Edge Detection",
                  f"source={source}",
                  f"f1={fmt_metric(metrics.get('f1'))}",
                  f"tpr={fmt_metric(metrics.get('tpr'))}",
                  f"tnr={fmt_metric(metrics.get('tnr'))}",
                  f"balanced_accuracy={fmt_metric(metrics.get('balanced_accuracy'))}",
                  f"precision={fmt_metric(metrics.get('precision'))}",
                  f"recall={fmt_metric(metrics.get('recall'))}",
                  f"tp={metrics.get('tp', 0)}",
                  f"fp={metrics.get('fp', 0)}",
                  f"fn={metrics.get('fn', 0)}",
                  f"tn={metrics.get('tn', 0)}",
                  f"positives={metrics.get('num_positive', 0)}",
                  f"predicted={metrics.get('num_predicted', 0)}",
            ]
            if extra:
                  parts.extend(f"{key}={value}" for key, value in extra.items())
            print(" | ".join(parts))


      def _print_tip_detection_metrics(self, tip, node_metrics, edge_metrics):
            parts = [
                  "Tip Detection",
                  f"tip={tip}",
                  f"node_f1={fmt_metric(node_metrics.get('f1'))}",
                  f"node_tpr={fmt_metric(node_metrics.get('tpr'))}",
                  f"node_tnr={fmt_metric(node_metrics.get('tnr'))}",
                  f"node_balanced_accuracy={fmt_metric(node_metrics.get('balanced_accuracy'))}",
                  f"node_precision={fmt_metric(node_metrics.get('precision'))}",
                  f"node_recall={fmt_metric(node_metrics.get('recall'))}",
                  f"edge_f1={fmt_metric(edge_metrics.get('f1'))}",
                  f"edge_tpr={fmt_metric(edge_metrics.get('tpr'))}",
                  f"edge_tnr={fmt_metric(edge_metrics.get('tnr'))}",
                  f"edge_balanced_accuracy={fmt_metric(edge_metrics.get('balanced_accuracy'))}",
                  f"edge_precision={fmt_metric(edge_metrics.get('precision'))}",
                  f"edge_recall={fmt_metric(edge_metrics.get('recall'))}",
                  f"node_tp={node_metrics.get('tp', 0)}",
                  f"node_fp={node_metrics.get('fp', 0)}",
                  f"node_fn={node_metrics.get('fn', 0)}",
                  f"edge_tp={edge_metrics.get('tp', 0)}",
                  f"edge_fp={edge_metrics.get('fp', 0)}",
                  f"edge_fn={edge_metrics.get('fn', 0)}",
            ]
            print(" | ".join(parts))


      def _report_edge_detection_metrics(self):
            ref = self.edge_detection_reference
            if ref is None:
                  return

            added_edges = ref["added_edges"]
            if len(added_edges) == 0:
                  print("Edge Detection | skipped=true | reason=no_added_pollution_edges")
                  return

            prompt_filter = getattr(self.prompt, "filter_module", None)
            if prompt_filter is not None:
                  filter_graph = SimpleNamespace(
                        x=self.data.x,
                        edge_index=self.data.edge_index,
                        num_nodes=self.data.num_nodes,
                  )
                  with torch.no_grad():
                        filter_output = prompt_filter(filter_graph)
                  filter_metrics = evaluate_edge_detection(
                        self.data.edge_index,
                        added_edges,
                        pred_anomaly_mask=~filter_output["edge_mask"],
                        anomaly_score=filter_output.get("edge_score", None),
                  )
                  self._print_edge_detection_metrics(
                        "filter_module",
                        filter_metrics,
                        {"mode": getattr(prompt_filter, "mode", self.filter_mode)},
                  )

            gppt_state = getattr(self.prompt, "last_filter_detection", None)
            if gppt_state is not None:
                  gppt_metrics = evaluate_edge_detection(
                        gppt_state["edge_index"],
                        added_edges,
                        pred_anomaly_mask=gppt_state.get("pred_anomaly_mask", None),
                        anomaly_score=gppt_state.get("edge_score", None),
                  )
                  self._print_edge_detection_metrics("gppt_filter", gppt_metrics)

            tau_state = getattr(self.prompt, "last_tau_detection", None)
            if tau_state is not None:
                  tau_metrics = evaluate_edge_detection(
                        tau_state["edge_index"],
                        added_edges,
                        pred_anomaly_mask=tau_state.get("pred_anomaly_mask", None),
                        anomaly_score=tau_state.get("edge_score", None),
                  )
                  self._print_edge_detection_metrics("tau_tune", tau_metrics)

            tip_state = getattr(self.prompt, "last_tip_detection", None)
            if tip_state is not None:
                  node_by_tip = tip_state.get("node_by_tip", {})
                  for tip in ["sim_pt", "degree_pt", "out_detect_pt"]:
                        if tip not in node_by_tip:
                              continue
                        node_metrics = evaluate_node_detection(
                              self.data.num_nodes,
                              added_edges,
                              node_by_tip.get(tip),
                        )
                        edge_metrics = evaluate_incident_edge_detection(
                              tip_state["edge_index"],
                              added_edges,
                              node_by_tip.get(tip),
                        )
                        self._print_tip_detection_metrics(tip, node_metrics, edge_metrics)

                  ood_mask = tip_state.get("out_detect_edge_mask", None)
                  if ood_mask is not None:
                        ood_metrics = evaluate_edge_detection(
                              tip_state["edge_index"],
                              added_edges,
                              pred_anomaly_mask=ood_mask,
                              anomaly_score=tip_state.get("out_detect_edge_score", None),
                        )
                        self._print_edge_detection_metrics("out_detect_pt_edges", ood_metrics)

                  




      def train(self, data):
            self.gnn.train()
            self.answering.train()
            self.optimizer.zero_grad() 
            out = self.gnn(data.x, data.edge_index, batch=None) 
            out = self.answering(out)
            loss = self.criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()  
            self.optimizer.step()  
            return loss.item()
 

      #####################################################################################################################################################
      # ↓
      # ↓
      # ↓
      def AllInOneTrainSynchro(self, train_loader):
            # 同时优化 发现效果都不如分开优化好，只tune answer头效果更好点
            # from torch import nn, optim
            # model_param_group = []
            # model_param_group.append({"params": self.prompt.parameters()})
            # model_param_group.append({"params": self.answering.parameters()})
            # AllInOne_opi = optim.Adam(model_param_group, lr=0.001, weight_decay=5e-4)
            
            # 只tune answer
            # AllInOne_opi = self.answer_opi
            
            # 只tune prompt
            AllInOne_opi = self.pg_opi

            self.prompt.train()
            self.answering.train()
            loss = self.prompt.Tune(train_loader, self.gnn, self.answering, self.criterion, AllInOne_opi, self.device)
            return loss
      
      def AllInOneTrain_Shield(self, train_loader, pseudo_logits_train):
             #we update answering and prompt alternately.
            
            answer_epoch = 20  # 50 80
            prompt_epoch = 20  # 50 80
      
            # tune task head
            self.answering.train()
            self.prompt.eval()
            for epoch in range(1, answer_epoch + 1):
                  answer_loss = self.prompt.TuneKnowledgeDistillation(train_loader, pseudo_logits_train, self.gnn,  self.answering, self.criterion, self.answer_opi, self.device)
                  print(("frozen gnn | frozen prompt | *tune answering function... {}/{} ,loss: {:.4f} ".format(epoch, answer_epoch, answer_loss)))

            # tune prompt
            self.answering.eval()
            self.prompt.train()
            for epoch in range(1, prompt_epoch + 1):
                  pg_loss = self.prompt.TuneKnowledgeDistillation(train_loader, pseudo_logits_train, self.gnn,  self.answering, self.criterion, self.pg_opi, self.device)
                  print(("frozen gnn | *tune prompt |frozen answering function... {}/{} ,loss: {:.4f} ".format(epoch, answer_epoch, pg_loss)))
            return pg_loss

      def AllInOneTrain(self, train_loader):
            #we update answering and prompt alternately.
            
            answer_epoch = 20  # 50 80
            prompt_epoch = 20  # 50 80
            
            # tune task head
            self.answering.train()
            self.prompt.eval()
            for epoch in range(1, answer_epoch + 1):
                  answer_loss = self.prompt.Tune(train_loader, self.gnn,  self.answering, self.criterion, self.answer_opi, self.device)
                  print(("frozen gnn | frozen prompt | *tune answering function... {}/{} ,loss: {:.4f} ".format(epoch, answer_epoch, answer_loss)))

            # tune prompt
            self.answering.eval()
            self.prompt.train()
            for epoch in range(1, prompt_epoch + 1):
                  pg_loss = self.prompt.Tune( train_loader,  self.gnn, self.answering, self.criterion, self.pg_opi, self.device)
                  print(("frozen gnn | *tune prompt |frozen answering function... {}/{} ,loss: {:.4f} ".format(epoch, answer_epoch, pg_loss)))
            return pg_loss
      # ↑
      # ↑
      # ↑
      #####################################################################################################################################################








      def MultiGpromptTrain(self, pretrain_embs, train_lbls, train_idx):
            self.DownPrompt.train()
            self.optimizer.zero_grad()
            prompt_feature = self.feature_prompt(self.features)
            # prompt_feature = self.feature_prompt(self.data.x)
            # embeds1 = self.gnn(prompt_feature, self.data.edge_index)
            embeds1= self.Preprompt.gcn(prompt_feature, self.sp_adj , True, False)
            pretrain_embs1 = embeds1[0, train_idx]
            # pretrain_embs  是使用preprompt的gcn生成的还没有加prompt的embs的train embs
            # pretrain_embs1 是加了提示后的train features生成的train embs
            logits = self.DownPrompt(pretrain_embs,pretrain_embs1, train_lbls, 1).float().to(self.device) # 1 shot  Cora torch.Size([7, 7])
            loss = self.criterion(logits, train_lbls)           
            loss.backward(retain_graph=True)
            self.optimizer.step()
            return loss.item()
      


      #####################################################################################################################################################
      # ↓
      # ↓
      # ↓
      def GPFTrain_Shield(self, train_loader, pseudo_logits_train): # 样本太少，蒸馏没什么意义
            self.prompt.train()
            self.answering.train()
            total_loss = 0.0 
            for batch in train_loader:  
                  self.optimizer.zero_grad() 
                  batch = batch.to(self.device)
                  batch.x = self.prompt.add(batch.x)
                  out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt = self.prompt, prompt_type = self.prompt_type)
                  out = self.answering(out)
                  # loss = self.criterion(out, batch.y)  
                  # loss_ce = self.criterion(out, torch.argmax(pseudo_logits_train, dim=1)) # 注意，这个有的时候预测不一定准确！在fewshot的时候可能还可以，但是shot多了貌似就不太行了，要具体分析一下
                  loss_ce = self.criterion(out, batch.y)  
                  # KL散度，知识蒸馏
                  temperature = 0.2
                  alpha = 0.1
                  pseudo_logits_train = pseudo_logits_train.detach()
                  loss_kl = torch.nn.KLDivLoss()(F.log_softmax(out / temperature, dim=1), F.softmax(pseudo_logits_train / temperature, dim=1)) 
                  loss = (1 - alpha) * loss_ce + alpha * loss_kl

                  loss.backward()  
                  self.optimizer.step()  
                  total_loss += loss.item()  
            return total_loss / len(train_loader) 


      def GPFTrain(self, train_loader):
            self.prompt.train()
            self.answering.train()
            total_loss = 0.0 
            for batch in train_loader:
                  self.optimizer.zero_grad() 
                  batch = batch.to(self.device)
                  batch.x = self.prompt.add(batch.x)
                  out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt = self.prompt, prompt_type = self.prompt_type)
                  out = self.answering(out)
                  # loss = self.criterion(out, batch.pseudo_label)  # 没有效果，看来inductive任务是不可行的，还是得从图处理的角度去考虑
                  loss = self.criterion(out, batch.y)  
                  loss.backward()  
                  self.optimizer.step()  
                  total_loss += loss.item()  
            return total_loss / len(train_loader) 
      

      def GPFTranductivetrain(self, data): 
            self.prompt.train()
            self.answering.train()

            # ################
            # # 放在前面: 先修剪图再对特征添加prompt
            # # Prune edge index
            # edge_index = data.edge_index
            # cosine_sim = F.cosine_similarity(data.x[edge_index[0]], data.x[edge_index[1]])
            # # Define threshold t
            # threshold = 0.6
            # # Identify edges to keep
            # keep_edges = cosine_sim >= threshold
            # # Filter edge_index to only keep edges above the threshold
            # pruned_edge_index = edge_index[:, keep_edges]
            # data  = Data(x=data.x, edge_index=pruned_edge_index, y=data.y, train_mask= data.train_mask, val_mask= data.val_mask, test_mask= data.test_mask)
            # prompted_x = self.prompt.add(data.x)
            # out = self.gnn(prompted_x, data.edge_index, prompt = self.prompt, prompt_type = self.prompt_type) # batch=None返回的是节点embedding
            # ################
  
            ###############
            # 前后都不处理，直接加提示
            prompted_x = self.prompt.add(data.x)
            out = self.gnn(prompted_x, data.edge_index, prompt = self.prompt, prompt_type = self.prompt_type) # batch=None返回的是节点embedding
            ###############

            # ################
            # # 放在后面： 先加提示后根据添加prompt的特征修剪图  threshold = 0.5 很好
            # prompted_x = self.prompt.add(data.x)
            # # Prune edge index
            # edge_index = data.edge_index
            # cosine_sim = F.cosine_similarity(prompted_x[edge_index[0]], prompted_x[edge_index[1]])
            # # Define threshold t
            # threshold = 0.5
            # # Identify edges to keep
            # keep_edges = cosine_sim >= threshold
            # # Filter edge_index to only keep edges above the threshold
            # pruned_edge_index = edge_index[:, keep_edges]
            # pruned_g  = Data(x=prompted_x, edge_index=pruned_edge_index, y=data.y)
            # out = self.gnn(prompted_x, pruned_g.edge_index)
            # ################


            out = self.answering(out)
            loss = self.criterion(out[data.train_mask], data.y[data.train_mask])
            self.optimizer.zero_grad() 
            loss.backward()  
            self.optimizer.step()
            return loss
      # ↑
      # ↑
      # ↑
      #####################################################################################################################################################



      def GPPTtrain(self, data):
            self.prompt.train()
            node_embedding = self.gnn(data.x, data.edge_index)
            out = self.prompt(node_embedding, data.edge_index)
            loss = self.criterion(out[data.train_mask], data.y[data.train_mask])
            loss = loss + 0.001 * constraint(self.device, self.prompt.get_TaskToken())
            self.pg_opi.zero_grad()
            loss.backward()
            self.pg_opi.step()
            mid_h = self.prompt.get_mid_h()
            self.prompt.update_StructureToken_weight(mid_h)
            return loss.item()



      def GpromptTrain(self, train_loader):
            self.prompt.train()
            total_loss = 0.0 
            accumulated_centers = None
            accumulated_counts = None
            for batch in train_loader:  
                  self.pg_opi.zero_grad() 
                  batch = batch.to(self.device)
                  out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt = self.prompt, prompt_type = 'Gprompt')

                  # out = s𝑡,𝑥 = ReadOut({p𝑡 ⊙ h𝑣 : 𝑣 ∈ 𝑉 (𝑆𝑥)}),
                  center, class_counts = center_embedding(out, batch.y, self.output_dim)
                   # 累积中心向量和样本数
                  if accumulated_centers is None:
                        accumulated_centers = center
                        accumulated_counts = class_counts
                  else:
                        accumulated_centers += center * class_counts
                        accumulated_counts += class_counts
                  criterion = Gprompt_tuning_loss()
                  loss = criterion(out, center, batch.y)  
                  loss.backward()  
                  self.pg_opi.step()  
                  total_loss += loss.item()
            # 计算加权平均中心向量
            mean_centers = accumulated_centers / accumulated_counts

            return total_loss / len(train_loader), mean_centers



      #####################################################################################################################################################
      # ↓
      # ↓
      # ↓
      def RobustPromptInductiveTrainSynchro(self, train_loader):
            self.prompt.train()
            self.answering.train()
            loss = self.prompt.Tune(train_loader, self.gnn, self.answering, self.criterion, self.optimizer, self.device)
            return loss

      # For RobustPrompt_I_Test
      #prompt和anwser头一起优化，为了使用知识蒸馏训练，维度对齐
      def RobustPromptInductiveTrain_KD(self, train_loader, pseudo_model, pseudo_logits_train):
            self.prompt.train()
            self.answering.train()
            loss = self.prompt.TuneKnowledgeDistillation(train_loader, pseudo_model, pseudo_logits_train, self.gnn,  self.answering, self.criterion, self.optimizer, self.device)
            return loss

      # For RobustPrompt_I_Test
      # prompt和anwser头分开优化
      def RobustPromptInductiveTrain(self, train_loader, remaining_loader, pseudo_model):
            #we update answering and prompt alternately.
            answer_epoch = 20  # 50 80
            prompt_epoch = 20  # 50 80
            
            # tune task head
            self.answering.train()
            self.prompt.eval()
            for epoch in range(1, answer_epoch + 1):
                  answer_loss = self.prompt.Tune(train_loader, remaining_loader, pseudo_model, self.gnn,  self.answering, self.criterion, self.answer_opi, self.device)
                  print(("frozen gnn | frozen prompt | *tune answering function... {}/{} ,loss: {:.4f} ".format(epoch, answer_epoch, answer_loss)))

            # tune prompt
            self.answering.eval()
            self.prompt.train()
            for epoch in range(1, prompt_epoch + 1):
                  pg_loss     = self.prompt.Tune(train_loader, remaining_loader, pseudo_model, self.gnn, self.answering, self.criterion, self.pg_opi, self.device)
                  print(("frozen gnn | *tune prompt |frozen answering function... {}/{} ,loss: {:.4f} ".format(epoch, answer_epoch, pg_loss)))
            
            return pg_loss
      

      def RobustPromptTranductivetrain(self, data): 
            self.prompt.train()
            self.answering.train()
            loss = self.prompt.Tune(data, self.gnn, self.answering, self.criterion, self.optimizer, self.device)
            return loss


      def RobustPromptTranductivetrain_PseudoLabels(self, data, idx_train_regenerate, pseudo_labels): #, iid_train, pruned_data, lambda_cmd, lambda_mse
            self.prompt.train()
            self.answering.train()
            self.optimizer.zero_grad() 
            # 这里要注意，和GPF不同，这里都是一个图而不是loader，所以一个epoch跑完后data.x就消失了，不能进行后向传播，要用一个新的值存储, 不能用data.x = self.prompt.add(data.x)，直接被覆盖，无法训练
            prompted_x = self.prompt.add(data.x)
            out       = self.gnn(prompted_x, data.edge_index, prompt = self.prompt, prompt_type = self.prompt_type)
            out = self.answering(out)
            # STRG 利用重新生成的idx_train和伪标签训练
            loss = self.criterion(out[idx_train_regenerate], pseudo_labels[idx_train_regenerate])
            loss.backward()  
            self.optimizer.step()
            return loss
      # ↑
      # ↑
      # ↑
      #####################################################################################################################################################













      def run(self):
            if self.prompt_type == 'MultiGprompt':
                  # 使用预训练的GCN得到还没有加prompt的embs
                  embeds, _ = self.Preprompt.embed(self.features, self.sp_adj, True, None, False)
                  idx_train  = self.data.train_mask.nonzero().squeeze()
                  train_lbls = self.data.y[self.data.train_mask].type(torch.long) 

                  idx_val    = self.data.val_mask.nonzero().squeeze()
                  val_lbls   = self.data.y[self.data.val_mask].type(torch.long)

                  idx_test    = self.data.test_mask.nonzero().squeeze()
                  test_lbls   = self.data.y[self.data.test_mask].type(torch.long)

                  pretrain_embs = embeds[0, idx_train].type(torch.long)
                  val_embs      = embeds[0, idx_val].type(torch.long)
                  test_embs     = embeds[0, idx_test].type(torch.long)





            if self.prompt_type in ['RobustPrompt-GPF','RobustPrompt-GPFplus','RobustPrompt-T-IA','RobustPrompt-T-NSP-IA']:
                  # 利用shot的标签训练一个pseudo label分类器
                  print("don't use structure")
                  idx_train  = self.data.train_mask.nonzero().squeeze().cpu()
                  idx_val    = self.data.val_mask.nonzero().squeeze().cpu()
                  idx_test    = self.data.test_mask.nonzero().squeeze().cpu()
                  # 效仿STRG首先利用train的标签训练一个不需要结构的MLP
                  n_hidden = 1024
                  batch_size = 64
                  weight_decay = 5e-4
                  lr = 1e-2
                  epochs = 200
                  loss = nn.CrossEntropyLoss()
                  k = 80 # 80
                  # dataloaders
                  pseudo_train_dataset = Data1.TensorDataset(self.data.x[idx_train], self.data.y[idx_train])
                  pseudo_train_loader = Data1.DataLoader(pseudo_train_dataset, batch_size=batch_size, shuffle=True)
                  pseudo_val_dataset = Data1.TensorDataset(self.data.x[idx_val], self.data.y[idx_val])
                  pseudo_val_loader = Data1.DataLoader(pseudo_val_dataset, batch_size=batch_size, shuffle=False)
                  pseudo_test_dataset = Data1.TensorDataset(self.data.x[idx_test], self.data.y[idx_test])
                  pseudo_test_loader = Data1.DataLoader(pseudo_test_dataset, batch_size=batch_size, shuffle=False)
                  pseudo_model = MLP(self.data.x.shape[1], self.output_dim, n_hidden)
                  pseudo_model = pseudo_model.to(self.device)
                  optimizer = torch.optim.Adam(pseudo_model.parameters(), lr=lr, weight_decay=weight_decay)
                  acc = train_MLP(pseudo_model, epochs, optimizer, pseudo_train_loader, pseudo_val_loader, pseudo_test_loader, loss, self.device)
                  print('Accuracy:%f' % acc)
                  print('Train Pseudo Model Done !')
                   # 扩展伪标签
                  logits = pseudo_model(self.data.x.to(self.device)).cpu()
                  pseudo_labels = self.data.y.clone()
                  idx_train_regenerate, pseudo_labels = get_psu_labels(logits, pseudo_labels, idx_train, idx_test, k=k, append_idx=True)
                  print(f'IA-PT: train nodes expanded {len(idx_train)} -> {len(idx_train_regenerate)}')

                  # 对 RobustPrompt-T-IA / RobustPrompt-T-NSP-IA：扩展 data.train_mask / data.y，让 Tune() 自动使用扩展标签
                  if self.prompt_type in ['RobustPrompt-T-IA', 'RobustPrompt-T-NSP-IA']:
                      self._saved_train_mask = self.data.train_mask.clone()
                      self._saved_y = self.data.y.clone()
                      self.data.train_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
                      self.data.train_mask[idx_train_regenerate] = True
                      self.data.y[idx_train_regenerate] = pseudo_labels[idx_train_regenerate]




            if self.prompt_type in []: 
                  from data_pyg.data_pyg import get_dataset
                  import os.path as osp
                  from torch_geometric.utils import remove_self_loops

                  path                     = attack_data_root()
                  clean_dataset_pretrain   = get_dataset(path, 'Attack-' + self.dataset_name, self.attack_method.split('-')[0], 0.0)
                  clean_data_pretrain      = clean_dataset_pretrain[0]
                  clean_data_pretrain.edge_index, _ = remove_self_loops(clean_data_pretrain.edge_index) # attack的时候不能有自环
                  clean_data_pretrain      = clean_data_pretrain.to(self.device)
                  gnn_copy = copy.deepcopy(self.gnn)
                  tune_answering_acc_test  = finetune_answering(gnn_copy, clean_data_pretrain, self.answering, self.criterion, self.output_dim, 300, self.device)
                  
                  # get the pseudo-labels, which will be used to train the detector
                  print("====== Get the pseudo-labels ======")
                  self.gnn.eval()
                  self.answering.eval()
                  out  =  self.gnn(clean_data_pretrain.x, clean_data_pretrain.edge_index, batch=None) 
                  pseudo_labels  =  self.answering(out).argmax(dim=1)
                  # print(sum(pseudo_labels == clean_data_pretrain.y) / len(clean_data_pretrain.y))

                  # Hyper-parameters for detector
                  d_epochs = 50
                  weight_decay_d = 1e-4
                  lr_d = 1e-2
                  loss_d = nn.BCEWithLogitsLoss()
                  batch_size = 2048
                  n_hidden_d = 64
                  dim_input = self.hid_dim * 2 + self.input_dim* 2
                  num_detectors = 5

                  print("====== Start training the detectors ======")
                  clean_dense_A = torch.zeros((clean_data_pretrain.x.shape[0], clean_data_pretrain.x.shape[0]), dtype=clean_data_pretrain.edge_index.dtype)
                  for i, (start, end) in enumerate(clean_data_pretrain.edge_index.t()):
                        clean_dense_A[start, end] = 1
                  clean_dense_A = torch.tensor(clean_dense_A, dtype=torch.float32).to(self.device)

                  idx_train = clean_data_pretrain.train_mask.nonzero().squeeze(-1)
                  idx_val = clean_data_pretrain.val_mask.nonzero().squeeze(-1)
                  idx_test = clean_data_pretrain.test_mask.nonzero().squeeze(-1)

                  detectors = []
                  for _ in range(num_detectors):
                        detector = MLP(dim_input, 1, n_hidden_d, n_layers=2).to(self.device)
                        optimizer_d = torch.optim.Adam(detector.parameters(), lr=lr_d, weight_decay=weight_decay_d)
                        detector = get_detector(detector, optimizer_d, d_epochs, loss_d, 0.2, clean_data_pretrain.x,
                                                clean_dense_A, clean_data_pretrain.y, pseudo_labels, self.device, idx_train, idx_val, idx_test, self.gnn, self.hid_dim,
                                                batch_size)
                        detectors.append(detector)


            # for all-in-one and Gprompt we use k-hop subgraph
            if self.prompt_type in ['All-in-one', 'Gprompt', 'GPF', 'GPF-plus','RobustPrompt-I', 'RobustPrompt-I-original']:

                  # print(len(self.train_dataset))
                  # print(len(self.val_dataset))
                  # print(len(self.test_dataset))

                  train_loader = DataLoader(self.train_dataset, batch_size=100, shuffle=True)
                  test_loader = DataLoader(self.test_dataset, batch_size=100, shuffle=False)
                  val_loader = DataLoader(self.val_dataset, batch_size=100, shuffle=False)
                  print("prepare induce graph data is finished!")

                  # print(len(train_loader))
                  # print(len(val_loader))
                  # print(len(test_loader))
                  # quit()


            print("run {}".format(self.prompt_type))

            ##########################################################################################################################################################
            # 训练方式一： 不用每个epoch都用验证集
            patience = 20
            best = 1e9
            cnt_wait = 0
            best_loss = 1e9
            batch_best_loss = []
            best_val_acc = final_test_acc = 0 # add by ssh 如果在训练中用验证集则加上，不用验证集可以取消

            for epoch in range(1, self.epochs + 1):
                  t0 = time.time()

                  if self.prompt_type  == 'None':
                        loss = self.train(self.data)       
                        # val_acc,  F1  = GNNNodeEva(self.data, self.data.val_mask, self.gnn, self.answering, self.output_dim, self.device)
                  elif self.prompt_type == 'All-in-one':
                        loss = self.AllInOneTrain(train_loader)    # train_loader
                        # val_acc, F1    = AllInOneEva(val_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)     
                  elif self.prompt_type == 'GPPT':
                        loss = self.GPPTtrain(self.data)    
                        # val_acc,  F1  = GPPTEva(self.data, self.data.val_mask, self.gnn, self.prompt, self.output_dim, self.device)      
                  elif self.prompt_type == 'Gprompt':
                        loss, center =  self.GpromptTrain(train_loader)
                        # val_acc, F1 = GpromptEva(val_loader, self.gnn, self.prompt, center, self.output_dim, self.device)
                  elif self.prompt_type in ['GPF', 'GPF-plus']:
                        loss = self.GPFTrain(train_loader) 
                        # val_acc, F1 = GPFEva(val_loader, self.gnn, self.prompt, self.answering, self.output_dim, self.device)                                             
                  elif self.prompt_type == 'MultiGprompt':
                        loss = self.MultiGpromptTrain(pretrain_embs, train_lbls, idx_train)
                        prompt_feature = self.feature_prompt(self.features)
                        # val_acc, F1 = MultiGpromptEva(val_embs, val_lbls, idx_val, prompt_feature, self.Preprompt, self.DownPrompt, self.sp_adj, self.output_dim, self.device)
                  elif self.prompt_type in ['GPF-Tranductive', 'GPF-plus-Tranductive']:
                        loss = self.GPFTranductivetrain(self.data)
                        # val_acc,  F1    = GPFTranductiveEva(self.data, self.data.val_mask, self.gnn, self.prompt, self.answering, self.output_dim, self.device)

                  # add by ssh
                  elif self.prompt_type in ['RobustPrompt-I', 'RobustPrompt-I-original']:
                        loss = self.RobustPromptInductiveTrainSynchro(train_loader)
                        # val_acc, F1    = RobustPromptInductiveEva(val_loader,  'Val',  pseudo_model, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
                  elif self.prompt_type in ['RobustPrompt-T', 'RobustPrompt-T-IA', 'RobustPrompt-T-NSP', 'RobustPrompt-T-NSP-IA']:
                        loss = self.RobustPromptTranductivetrain(self.data)
                        # test_acc, F1    = RobustPromptTranductiveEva(self.data, self.data.val_mask,  self.gnn, self.prompt, self.answering, self.output_dim, self.device)
                  elif self.prompt_type in ['RobustPrompt-GPF', 'RobustPrompt-GPFplus']:
                        loss = self.RobustPromptTranductivetrain_PseudoLabels(self.data, idx_train_regenerate, pseudo_labels)
                        # val_acc,  F1    = RobustPromptTranductiveEva(self.data, self.data.val_mask,   self.gnn, self.prompt, self.answering, self.output_dim, self.device)

                  if loss < best:
                        best = loss
                        # best_t = epoch
                        cnt_wait = 0
                        # torch.save(model.state_dict(), args.save_name)
                  else:
                        cnt_wait += 1
                        if cnt_wait == patience:
                              print('-' * 100)
                              print('Early stopping at '+str(epoch) +' eopch!')
                              break
                  print("Epoch {:03d} |  Time(s) {:.4f} | {} Loss {:.4f}  ".format(epoch, time.time() - t0, self.prompt_type, loss))

            # print(self.data)
            # print('change eva data!')
            # from data_pyg.data_pyg import get_dataset
            # import os.path as osp
            # path      = osp.expanduser('/home/songsh/MyPrompt/data_pyg/Attack_data')
            # self.data = get_dataset(path, 'Attack-' + self.dataset_name, 'random', '0.5')[0]
            # self.data = self.data.to(self.device) 
            # print(self.data)

            
            # IA-PT: 训练完成，恢复原始 train_mask / y，确保评估使用真实标签
            if hasattr(self, '_saved_train_mask') and self._saved_train_mask is not None:
                self.data.train_mask = self._saved_train_mask
                self.data.y = self._saved_y
                self._saved_train_mask = None
                self._saved_y = None

            import math
            test_acc = float('nan')
            if not math.isnan(loss):
                  batch_best_loss.append(loss)
                  if self.prompt_type == 'None':
                        test_acc, F1   = GNNNodeEva(self.data, self.data.test_mask, self.gnn, self.answering,self.output_dim, self.device)          
                  elif self.prompt_type == 'All-in-one':
                        test_acc, F1   = AllInOneEva(test_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
                        # test_acc, F1   = AllInOneEva(test_loader, self.prompt, self.gnn, self.answering, self.output_dim, detectors,self.device)
                  elif self.prompt_type == 'GPPT':
                        test_acc, F1           = GPPTEva(self.data, self.data.test_mask, self.gnn, self.prompt, self.output_dim, self.device)
                  elif self.prompt_type =='Gprompt':
                        test_acc, F1           = GpromptEva(test_loader, self.gnn, self.prompt, center, self.output_dim, self.device)
                  elif self.prompt_type in ['GPF', 'GPF-plus']:
                        test_acc, F1 = GPFEva(test_loader, self.gnn, self.prompt, self.answering, self.output_dim, self.device)     
                        # test_acc, F1 = GPFEva(test_loader, self.gnn, self.prompt, self.answering, self.output_dim, detectors,self.device)                                         
                  elif self.prompt_type == 'MultiGprompt':
                        prompt_feature  = self.feature_prompt(self.features)
                        test_acc, F1    = MultiGpromptEva(test_embs, test_lbls, idx_test, prompt_feature, self.Preprompt, self.DownPrompt, self.sp_adj, self.output_dim, self.device)
                  elif self.prompt_type in ['GPF-Tranductive', 'GPF-plus-Tranductive']:
                        test_acc, F1    = GPFTranductiveEva(self.data, self.data.test_mask,  self.gnn, self.prompt, self.answering, self.output_dim, self.device)
            
                  # add by ssh
                  elif self.prompt_type in ['RobustPrompt-I', 'RobustPrompt-I-original']:
                        test_acc, F1    = RobustPromptInductiveEva(test_loader, self.gnn, self.prompt, self.answering, self.output_dim, self.device)
                  elif self.prompt_type in ['RobustPrompt-T', 'RobustPrompt-T-IA', 'RobustPrompt-T-NSP', 'RobustPrompt-T-NSP-IA']:
                        test_acc, F1    = RobustPromptTranductiveEva(self.data, self.data.test_mask,  self.gnn, self.prompt, self.answering, self.output_dim, self.device)
                  elif self.prompt_type in ['RobustPrompt-GPF', 'RobustPrompt-GPFplus']:
                        # 直接用GPF的评估方式就行
                        test_acc, F1    = GPFTranductiveEva(self.data, self.data.test_mask,  self.gnn, self.prompt, self.answering, self.output_dim, self.device)

                  # print(f"Final True Accuracy: {test_acc:.4f} | Macro F1 Score: {f1:.4f} | AUROC: {roc:.4f} | AUPRC: {prc:.4f}" )
                  print(f"Final True Accuracy: {test_acc:.4f} | Macro F1 Score: {F1:.4f}" )
                  self._report_edge_detection_metrics()
                  print("best_loss",  batch_best_loss)     
            return test_acc.cpu().numpy() if isinstance(test_acc, torch.Tensor) else test_acc






            # ##########################################################################################################################################################
            # # 训练方式二：每个epoch都用验证集 效率低
            # best_val_acc = final_test_acc = 0
            # # 用tqdm 更简洁
            # # pbar = tqdm(range(0, self.epochs))
            # # for epoch in pbar:
            # for epoch in range(0, self.epochs):
            #       t0 = time.time()
            #       if self.prompt_type == 'None':
            #             loss = self.train(self.data)
            #             print("Train Done!")
            #             val_acc,  F1  = GNNNodeEva(self.data, self.data.val_mask, self.gnn, self.answering, self.output_dim, self.device)
            #             print("Val Done!")
            #             test_acc, F1  = GNNNodeEva(self.data, self.data.test_mask, self.gnn, self.answering, self.output_dim, self.device)
            #             print("Test Done!")
                         
            #       elif self.prompt_type == 'All-in-one':
            #             # print("run All-in-one Prompt")
            #             loss = self.AllInOneTrain(train_loader)
            #             # 看下训练集的训练情况，是不是在被攻击数据上过拟合了 不用的话就注释掉
            #             train_acc, F1  = AllInOneEva(train_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
            #             print("train batch Done!")
            #             val_acc, F1    = AllInOneEva(val_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
            #             print("val batch Done!")
            #             test_acc, F1   = AllInOneEva(test_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
            #             print("test batch Done!")
                  
            #       elif self.prompt_type == 'GPPT':
            #             print("run GPPT Prompt")
            #             loss     = self.GPPTtrain(self.data)
            #             train_acc,  F1  = GPPTEva(self.data, self.data.train_mask, self.gnn, self.prompt, self.output_dim, self.device)
            #             print("Train Done!")
            #             val_acc,  F1  = GPPTEva(self.data, self.data.val_mask, self.gnn, self.prompt, self.output_dim, self.device)
            #             print("Val Done!")
            #             test_acc, F1  = GPPTEva(self.data, self.data.test_mask, self.gnn, self.prompt, self.output_dim, self.device)
            #             print("Test Done!")

            #       elif self.prompt_type =='Gprompt':
            #             print("run Graph Prompt")
            #             loss, center =  self.GpromptTrain(train_loader)
            #             train_acc, F1 = GpromptEva(train_loader, self.gnn, self.prompt, center, self.output_dim, self.device)
            #             print("train batch Done!")
            #             val_acc, F1 = GpromptEva(val_loader, self.gnn, self.prompt, center, self.output_dim, self.device)
            #             print("val batch Done!")
            #             test_acc, F1= GpromptEva(test_loader, self.gnn, self.prompt, center, self.output_dim, self.device)
            #             print("test batch Done!")


            #       elif self.prompt_type in ['GPF', 'GPF-plus']:
            #             print("run GPF/GPF-Plus Prompt")
            #             loss = self.GPFTrain(train_loader)
            #             train_acc, F1 = GPFEva(train_loader, self.gnn, self.prompt, self.answering, self.output_dim, self.device)    
            #             print("train batch Done!")
            #             val_acc, F1 = GPFEva(val_loader, self.gnn, self.prompt, self.answering, self.output_dim, self.device)    
            #             print("val batch Done!")
            #             test_acc, F1 = GPFEva(test_loader, self.gnn, self.prompt, self.answering, self.output_dim, self.device)    
            #             print("test batch Done!")


            #       elif self.prompt_type == 'MultiGprompt':
            #             print("run MultiGprompt")
            #             loss = self.MultiGpromptTrain(pretrain_embs, train_lbls, idx_train)
   
            #             # 记得 open prompt
            #             prompt_feature = self.feature_prompt(self.features)
            #             train_acc, F1 = MultiGpromptEva(pretrain_embs, train_lbls, idx_train, prompt_feature, self.Preprompt, self.DownPrompt, self.sp_adj, self.output_dim, self.device)
            #             print("Train Done!")
            #             # 记得 open prompt
            #             prompt_feature = self.feature_prompt(self.features)
            #             val_acc, F1 = MultiGpromptEva(val_embs, val_lbls, idx_val, prompt_feature, self.Preprompt, self.DownPrompt, self.sp_adj, self.output_dim, self.device)
            #             print("Val Done!")
            #             # 记得 open prompt
            #             prompt_feature = self.feature_prompt(self.features)
            #             test_acc, F1 = MultiGpromptEva(test_embs, test_lbls, idx_test, prompt_feature, self.Preprompt, self.DownPrompt, self.sp_adj, self.output_dim, self.device)
            #             print("Test Done!")


            #       if val_acc > best_val_acc:
            #             best_val_acc = val_acc
            #             final_test_acc = test_acc
            #       # print("Epoch {:03d} |  Time(s) {:.4f} | Loss {:.4f} | val Accuracy {:.4f} | test Accuracy {:.4f} ".format(epoch + 1, time.time() - t0, loss, val_acc, test_acc)) 
                  
            #       # 看下训练集的训练情况，是不是在被攻击数据上过拟合了
            #       # 果然 Epoch 009 |  Time(s) 5.1142 | Loss 3.3146 | train Accuracy 0.7143 | val Accuracy 0.3429 | test Accuracy 0.3059
            #       print("Epoch {:03d} |  Time(s) {:.4f} | Loss {:.4f} | train Accuracy {:.4f} | val Accuracy {:.4f} | test Accuracy {:.4f} ".format(epoch + 1, time.time() - t0, loss, train_acc, val_acc, test_acc))       

            #       # 使用tqdm进行显示 更简洁
            #       # pbar.set_description("Epoch {:03d} |  Time(s) {:.4f} | Loss {:.4f} | train Accuracy {:.4f} | val Accuracy {:.4f} | test Accuracy {:.4f} ".format(epoch + 1, time.time() - t0, loss, train_acc, val_acc, test_acc))       


            # print(f'Final Test: {final_test_acc:.4f}')
            # print("Node Task completed")
            # return final_test_acc.cpu().numpy() if isinstance(final_test_acc, torch.Tensor) else final_test_acc
            # ##########################################################################################################################################################





