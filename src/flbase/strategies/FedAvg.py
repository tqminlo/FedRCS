from collections import OrderedDict, Counter
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch

try:
    import wandb
except ModuleNotFoundError:
    pass
from ..server import Server
from ..client import Client
from ..models.CNN import *
from ..models.MLP import *
from ..utils import setup_optimizer, linear_combination_state_dict, setup_seed
from ...utils import autoassign, save_to_pkl, access_last_added_element
import time
import torch

from torch.utils.data import DataLoader

from itertools import product
from scipy.cluster.hierarchy import fcluster, linkage

class FedAvgClient(Client):
    def __init__(self, criterion, trainset, testset, client_config, cid, device, **kwargs):
        super().__init__(criterion, trainset, testset,
                         client_config, cid, device, **kwargs)
        self._initialize_model()

        temp = [self.count_by_class[cls] if cls in self.count_by_class.keys() else 1e-12 for cls in
                range(client_config['num_classes'])]
        self.count_by_class_full = torch.tensor(temp).to(self.device)
        self.D_client = sum([self.count_by_class[cls] for cls in self.count_by_class.keys()])
        # print(np.array(temp).astype(int).tolist(), ",")  # dem so mau moi class trong moi client
        # print("*** CHECK DATA *** : ", np.sum(np.array(temp).astype(int)))  # dem so mau moi client

        self.p = np.array(temp) / np.sum(np.array(temp))
        self.entropy = - np.sum(self.p * (np.log(self.p)))
        self.score = np.sum(np.array(temp)) * self.entropy
        # print(self.entropy)
        print(self.score)
        self.loss_utility = 0

    def _initialize_model(self):
        # parse the model from config file
        self.model = eval(self.client_config['model'])(self.client_config).to(self.device)
        # this is needed if the criterion has stateful tensors.
        self.criterion = self.criterion.to(self.device)

    def training(self, round, num_epochs):
        """
            Note that in order to use the latest server side model the `set_params` method should be called before `training` method.
        """
        setup_seed(round + self.client_config['global_seed'])
        # train mode
        self.model.train()
        # tracking stats
        self.num_rounds_particiapted += 1
        loss_seq = []
        acc_seq = []
        if self.trainloader is None:
            raise ValueError("No trainloader is provided!")
        optimizer = setup_optimizer(self.model, self.client_config, round)
        # print('lr:', optimizer.param_groups[0]['lr'])
        loss_utility = 0
        for i in range(num_epochs):
            epoch_loss, correct = 0.0, 0
            for j, (x, y) in enumerate(self.trainloader):
                # if len(y) == 1:
                #     continue
                # forward pass
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model.forward(x)
                loss = self.criterion(yhat, y)
                # backward pass
                # model.zero_grad safer and memory-efficient
                self.model.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=filter(lambda p: p.requires_grad, self.model.parameters()),
                                               max_norm=10)
                optimizer.step()
                predicted = yhat.data.max(1)[1]
                correct += predicted.eq(y.data).sum().item()
                epoch_loss += loss.item() * x.shape[0]  # rescale to bacthsize
                loss_utility += (loss.item() ** 2) * len(y)
            epoch_loss /= len(self.trainloader.dataset)
            epoch_accuracy = correct / len(self.trainloader.dataset)
            loss_seq.append(epoch_loss)
            acc_seq.append(epoch_accuracy)
        self.new_state_dict = self.model.state_dict()
        self.train_loss_dict[round] = loss_seq
        self.train_acc_dict[round] = acc_seq

        self.loss_utility = loss_utility / num_epochs

    def upload(self):
        return self.new_state_dict, self.count_by_class_full, (self.entropy, self.score)

    def testing(self, round, testloader=None):
        self.model.eval()
        if testloader is None:
            testloader = self.testloader
        test_count_per_class = Counter(testloader.dataset.targets.numpy())
        num_classes = self.client_config['num_classes']
        test_count_per_class = torch.tensor([test_count_per_class[cls] * 1.0 for cls in range(num_classes)])
        test_correct_per_class = torch.tensor([0] * num_classes)
        test_TN_per_class = torch.tensor([0] * num_classes)

        weight_per_class_dict = {'uniform': torch.tensor([1.0] * num_classes),
                                 'validclass': torch.tensor([0.0] * num_classes),
                                 'labeldist': torch.tensor([0.0] * num_classes)}
        for cls in self.label_dist.keys():
            # print("---", cls, type(cls))
            weight_per_class_dict['labeldist'][cls] = self.label_dist[cls]
            weight_per_class_dict['validclass'][cls] = 1.0
        # start testing
        predict_per_class = [0 for i in range(num_classes)]
        prob_per_class = torch.tensor([0 for i in range(num_classes)], device=self.device)
        with torch.no_grad():
            for i, (x, y) in enumerate(testloader):
                # forward pass
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model.forward(x)
                # stats
                predicted = yhat.data.max(1)[1]
                prob_per_class_batch = torch.mean(torch.nn.Softmax(dim=-1)(yhat), dim=0)
                prob_per_class = (prob_per_class * i + prob_per_class_batch) / (i+1)
                #######classes_shown_in_this_batch = torch.unique(y).cpu().numpy()
                #######for cls in classes_shown_in_this_batch:
                for cls in range(num_classes):
                    test_correct_per_class[cls] += ((predicted == y) * (y == cls)).sum().item()
                    test_TN_per_class[cls] += ((predicted != cls) * (y != cls)).sum().item()
                    predict_per_class[cls] += (predicted == cls).sum().item()
        acc_by_critertia_dict = {}
        for k in weight_per_class_dict.keys():
            acc_by_critertia_dict[k] = (((weight_per_class_dict[k] * test_correct_per_class).sum()) /
                                        ((weight_per_class_dict[k] * test_count_per_class).sum())).item()

        acc_by_critertia_dict["BACC"] = ((test_correct_per_class / test_count_per_class +
                                          test_TN_per_class / (
                                                  test_count_per_class.sum() - test_count_per_class)) / 2).mean()

        self.test_acc_dict[round] = {'acc_by_criteria': acc_by_critertia_dict,
                                     'correct_per_class': test_correct_per_class,
                                     'weight_per_class': weight_per_class_dict,
                                     'predict_per_class': predict_per_class,
                                     'prob_per_class': prob_per_class}


class FedAvgServer(Server):
    def __init__(self, server_config, clients_dict, exclude, cs_method, **kwargs):
        super().__init__(server_config, clients_dict, **kwargs)
        self.clients_dict = clients_dict
        self.summary_setup()
        # print("----------")
        # param = [tens.detach().to("cpu").flatten() for tens in list(self.clients_dict[0].model.parameters())]
        # param = np.concatenate(param)
        # print(param, len(param))
        self.server_model_state_dict = deepcopy(self.clients_dict[0].get_params())
        # make sure the starting point is correct
        self.server_side_client.set_params(self.server_model_state_dict, exclude_keys=set())
        # print(self.server_side_client.model.parameters())
        self.exclude_layer_keys = set()
        for key in self.server_model_state_dict:
            for ekey in exclude:
                if ekey in key:
                    self.exclude_layer_keys.add(key)
        if len(self.exclude_layer_keys) > 0:
            print(f"{self.server_config['strategy']}Server: the following keys will not be aggregated:\n ",
                  self.exclude_layer_keys)
        # freeze_layers = []
        # for param in self.server_side_client.model.named_parameters():
        #     if param[1].requires_grad == False:
        #         freeze_layers.append(param[0])
        # if len(freeze_layers) > 0:
        #     print("{self.server_config['strategy']}Server: the following layers will not be updated:", freeze_layers)
        self.cs_method = cs_method
        self.validloader = DataLoader(kwargs["global_validset"], batch_size=128, shuffle=False) if kwargs[
            "global_validset"] else None
        self.active_clients_indicies = []

        # for FedMCS/FedRCS methods
        self.sort_client_entropy = np.zeros(shape=(10, 10), dtype=int)
        self.sort_client_ref = np.arange(100,).reshape((10, 10))
        self.sort_client = np.arange(100,).reshape((10, 10))

        # for Cluster1 method
        self.cluster1_setup(clients_dict)

        # for OORT method
        self.n_all_clients = len(self.clients_dict.keys())      # = self.server_config["num_clients"]
        self.n_selected = int(self.n_all_clients * self.server_config["participate_ratio"])  # 10
        self.client_utilities = {client_id: 0 for client_id in range(0, self.n_all_clients)}
        self.client_last_rounds = {client_id: 0 for client_id in range(0, self.n_all_clients)}
        self.client_selected_times = {client_id: 0 for client_id in range(0, self.n_all_clients)}
        self.unexplored_clients = list(range(0, self.n_all_clients))
        self.current_round = 0
        self.blacklist_num = 40 # 2 x times of the average (2 x 20)
        self.blacklist = []
        self.cut_off = 0.95

        # for TQM2 method
        self.pre_global_model = deepcopy(self.server_side_client.model)

    def ranking(self, client_uploads, round):
        res = round % 55
        assert 1 <= res <= 10, "false round"

        D_list = []
        entropy_list = []
        ref_list = []
        for idx, (_, count_per_class_client, ref_score) in enumerate(client_uploads):
            num_data_in_client = torch.sum(count_per_class_client)
            D_list.append(num_data_in_client)
            entropy_list.append(ref_score[0])
            ref_list.append(ref_score[1])

        mul_acc_list = []
        for idx, (client_state_dict, _, _) in enumerate(client_uploads):
            count_per_class_test = Counter(self.server_side_client.testloader.dataset.targets.numpy())
            num_classes = self.server_side_client.client_config['num_classes']
            count_per_class_test = torch.tensor([count_per_class_test[cls] * 1.0 for cls in range(num_classes)])
            self.server_side_client.set_params(client_state_dict, self.exclude_layer_keys)
            self.server_side_client.testing(round, testloader=None)  # for Cifar10 exps, if use proxy set: testloader=self.validloader
            self.gfl_test_acc_dict[round] = self.server_side_client.test_acc_dict[round]
            acc_per_class = self.gfl_test_acc_dict[round]['correct_per_class']
            # print("---check acc_per_class : ", acc_per_class)
            mul_acc = torch.prod(np.exp(acc_per_class / count_per_class_test))
            mul_acc_list.append(mul_acc.cpu().numpy())

        entropy_pseudo = []
        for idx, (client_state_dict, _, _) in enumerate(client_uploads):
            self.server_side_client.set_params(client_state_dict, self.exclude_layer_keys)
            self.server_side_client.testing(round, testloader=self.validloader) # use as a pseudo, not valid
            self.gfl_test_acc_dict[round] = self.server_side_client.test_acc_dict[round]
            # predict_per_class = self.gfl_test_acc_dict[round]['predict_per_class']
            # print("---check predict_per_class : ", predict_per_class)
            # # p = np.array(predict_per_class) / np.sum(np.array(predict_per_class)) + 1e-12
            #             # # entropy = - np.sum(p * (np.log(p)))
            #             # num_data_noise = sum(predict_per_class)
            #             # num_classes = self.server_side_client.client_config['num_classes']
            #             # num_noise_per_class = num_data_noise//num_classes
            #             # p = np.clip(predict_per_class, 0, num_noise_per_class)
            #             # entropy = np.prod(np.exp(p / num_noise_per_class))
            prob_per_class = self.gfl_test_acc_dict[round]['prob_per_class'].cpu().numpy()
            print("---check prob_per_class : ", prob_per_class)
            entropy = np.prod(np.exp(prob_per_class))
            entropy_pseudo.append(entropy)

        # Calculate gradient list
        gradients_list = []
        for idx, (client_state_dict, count_per_class_client, _) in enumerate(client_uploads):
            gradient = []
            for state_key in client_state_dict.keys():
                if state_key not in self.exclude_layer_keys:
                    g = client_state_dict[state_key] - self.server_model_state_dict[state_key]
                    gradient.append(g.cpu().numpy().flatten())
            gradient = np.concatenate(gradient, axis=0)
            gradients_list.append(gradient)

        weights = torch.tensor([1] * 10)
        # weights = torch.tensor(D_list)
        weights = weights / torch.sum(weights)
        weights_2d = weights.unsqueeze(1)
        gradients_list = torch.tensor(gradients_list)
        gradient_global = gradients_list * weights_2d
        gradient_global = torch.sum(gradient_global, 0)
        cosin_similary_list = [torch.nn.CosineSimilarity(dim=0)(gradient_global, grad_k) for grad_k in
                               gradients_list]

        # Sort
        # D_list = torch.tensor(np.array(D_list))
        mul_acc_list = torch.tensor(np.array(mul_acc_list))

        if self.cs_method == "FedMCSdg":
            score_final = [D_list[i] * cosin_similary_list[i] for i in range(len(mul_acc_list))]
        elif self.cs_method == "FedMCSag":
            score_final = [mul_acc_list[i] * cosin_similary_list[i] for i in range(len(mul_acc_list))]
        elif self.cs_method == "FedMCSda":
            score_final = [D_list[i] * mul_acc_list[i] for i in range(len(mul_acc_list))]
        elif self.cs_method == "FedMCS*":
            score_final = [ref_list[i] for i in range(len(mul_acc_list))]
        elif self.cs_method == "FedRCS":
            score_final = [D_list[i] * entropy_pseudo[i] * cosin_similary_list[i] for i in range(len(mul_acc_list))]
        else:   # FedMCS
            score_final = [D_list[i] * mul_acc_list[i] * cosin_similary_list[i] for i in range(len(mul_acc_list))]

        print("---check active_clients_indicies : ", self.active_clients_indicies)
        sorted_idx = np.array([x for _, x in sorted(zip(score_final, self.active_clients_indicies), reverse=True)])
        print("---check sorted_idx_score : ", sorted_idx)
        self.sort_client[res - 1] = sorted_idx
        print("---check active_clients_indicies : ", self.active_clients_indicies)

        print("---check ref_list : ", ref_list)
        sorted_ref = np.array([x for _, x in sorted(zip(ref_list, self.active_clients_indicies), reverse=True)])
        print("---check sorted_ref : ", sorted_ref)
        self.sort_client_ref[res - 1] = sorted_ref

        if res == 10:
            self.sort_client = self.sort_client.transpose()
            print("---check self.sort_client : ")
            print(self.sort_client)

            self.sort_client_ref = self.sort_client_ref.transpose()
            print("---check self.sort_client_ref : ")
            print(self.sort_client_ref)

    def aggregate(self, client_uploads, round):
        num_participants = len(client_uploads)
        update_direction_state_dict = None
        exclude_layer_keys = self.exclude_layer_keys
        print("-----------", exclude_layer_keys)

        with torch.no_grad():
            for idx, (client_state_dict, count_per_class_client, _) in enumerate(client_uploads):
                client_update = linear_combination_state_dict(client_state_dict,
                                                              self.server_model_state_dict,
                                                              1.0,
                                                              -1.0,
                                                              )
                # num_data_in_client = np.sum(count_per_class_client.cpu().numpy())
                if idx == 0:
                    update_direction_state_dict = client_update
                    # update_direction_state_dict = linear_combination_state_dict(client_update, client_update, 0,
                    #                                                             num_data_in_client,
                    #                                                             )
                    # sum_samples_participants = num_data_in_client
                else:
                    update_direction_state_dict = linear_combination_state_dict(update_direction_state_dict,
                                                                                client_update,
                                                                                1.0,
                                                                                1.0,
                                                                                # num_data_in_client,
                                                                                )
                    # sum_samples_participants += num_data_in_client

            self.server_model_state_dict = linear_combination_state_dict(self.server_model_state_dict,
                                                                         update_direction_state_dict,
                                                                         1.0,
                                                                         1 / num_participants,
                                                                         # 1 / sum_samples_participants,
                                                                         )

    def testing(self, round, active_only=True, **kwargs):
        """
        active_only: only compute statiscs with to the active clients only
        """
        # get the latest global model
        self.server_side_client.set_params(self.server_model_state_dict, self.exclude_layer_keys)

        # test the performance for global models
        self.server_side_client.testing(round, testloader=None)  # use global testdataset
        print(' server global model correct',
              torch.sum(self.server_side_client.test_acc_dict[round]['correct_per_class']).item())

    def collect_stats(self, stage, round, active_only, **kwargs):
        """
            No actual training and testing is performed. Just collect stats.
            stage: str;
                {"train", "test"}
            active_only: bool;
                True: compute stats on active clients only
                False: compute stats on all clients
        """
        # get client_indices
        client_indices = self.clients_dict.keys()
        if active_only:
            client_indices = self.active_clients_indicies
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        if stage == 'train':
            for cid in client_indices:
                client = self.clients_dict[cid]
                # client.train_loss_dict[round] is a list compose the training loss per end of each epoch
                loss, acc, num_samples = client.train_loss_dict[round][-1], client.train_acc_dict[round][
                    -1], client.num_train_samples
                total_loss += loss * num_samples
                total_acc += acc * num_samples
                total_samples += num_samples
            average_loss, average_acc = total_loss / total_samples, total_acc / total_samples
            self.average_train_loss_dict[round] = average_loss
            self.average_train_acc_dict[round] = average_acc
        else:
            # test stage
            # get global model performance
            self.gfl_test_acc_dict[round] = self.server_side_client.test_acc_dict[round]
            acc_criteria = self.server_side_client.test_acc_dict[round]['acc_by_criteria'].keys()
            # get local model average performance
            # self.average_pfl_test_acc_dict[round] = {key: 0.0 for key in acc_criteria}
            # for cid in client_indices:
            #     client = self.clients_dict[cid]
            #     acc_by_criteria_dict = client.test_acc_dict[round]['acc_by_criteria']
            #     for key in acc_criteria:
            #         self.average_pfl_test_acc_dict[round][key] += acc_by_criteria_dict[key]

            # num_participants = len(client_indices)
            # for key in acc_criteria:
            #     self.average_pfl_test_acc_dict[round][key] /= num_participants

    def run(self, **kwargs):
        if self.server_config['use_tqdm']:
            round_iterator = tqdm(range(self.rounds + 1, self.server_config['num_rounds'] + 1), desc="Round Progress")
        else:
            round_iterator = range(self.rounds + 1, self.server_config['num_rounds'] + 1)

        # round index begin with 1
        best_test_acc = 0
        for r in round_iterator:
            setup_seed(r + kwargs['global_seed'])
            self.current_round = r

            if self.cs_method == "Random":
                self.active_clients_indicies = self.select_clients(self.server_config['participate_ratio'])

            elif self.cs_method == "FedMCSv1":
                if r <= 10:
                    self.active_clients_indicies = np.arange((r - 1) * 10, r * 10)
                elif r <= 10 + 5:
                    self.active_clients_indicies = deepcopy(self.sort_client[0])
                elif 10 + 5 < r <= 10 + 5 + 15 * 3:
                    rr = (r - 11 - kwargs["T21"]) % 15
                    k = int(np.abs(np.sqrt(2 * (rr + 1) + 1 / 4) - 1 / 2))
                    residual = int((rr + 1) - k * (k + 1) / 2)
                    if residual == 1:
                        clients_consider = deepcopy(self.sort_client[:k + 1]).flatten()
                        np.random.shuffle(clients_consider)
                        print("round :", r, "k :", k, "res :", residual)
                        print("client consider :", clients_consider)
                    elif residual == 0 and k == 1:
                        clients_consider = deepcopy(self.sort_client[0])
                        print("client consider :", clients_consider)
                    self.active_clients_indicies = clients_consider[10 * (
                            residual - 1):10 * residual] if residual > 0 else clients_consider[-10:]
                else:
                    self.active_clients_indicies = self.select_clients(self.server_config['participate_ratio'])

            elif self.cs_method in ["FedMCS", "FedMCS*", "FedMCSda", "FedMCSdg", "FedMCSag", "FedRCS"]:
                if r <= 10:
                    self.active_clients_indicies = deepcopy(self.sort_client[r-1])
                else:
                    rr = (r-11) % 55
                    k = int(np.abs(np.sqrt(2 * (rr + 1) + 1 / 4) - 1 / 2))
                    residual = int((rr + 1) - k * (k + 1) / 2)
                    if residual == 1:
                        clients_consider = deepcopy(self.sort_client[:k + 1]).flatten()
                        np.random.shuffle(clients_consider)
                        print("round :", r, "k :", k, "res :", residual)
                        print("client consider :", clients_consider)
                    elif residual == 0 and k == 1:
                        clients_consider = deepcopy(self.sort_client[0])
                        print("client consider :", clients_consider)
                    self.active_clients_indicies = clients_consider[10 * (
                            residual - 1):10 * residual] if residual > 0 else clients_consider[-10:]

            elif self.cs_method == "Cluster1":
                self.active_clients_indicies = np.zeros(shape=(self.n_cluster,))
                all_idx = np.arange(self.server_config["num_clients"])  # (0, 1, ..., 99)
                for k in range(self.n_cluster):
                    weights = self.distri_clusters[k]
                    self.active_clients_indicies[k] = int(np.random.choice(all_idx, 1, p=weights / sum(weights)))

            elif self.cs_method == "Cluster2":
                self.active_clients_indicies = self.cluster2_cs()

            elif self.cs_method == "OORT":
                self.active_clients_indicies = self.oort_cs()

            elif self.cs_method == "OORT2":
                self.active_clients_indicies = self.oort2_cs()

            elif self.cs_method == "TQM":
                self.active_clients_indicies = self.tqm_cs()

            elif self.cs_method == "TQM2":
                self.active_clients_indicies = self.tqm2_cs()

            elif self.cs_method == "MOORT":
                self.active_clients_indicies = self.moort_cs()

            # active clients download weights from the server
            tqdm.write(f"Round:{r} - Active clients:{self.active_clients_indicies}:")

            for cid in self.active_clients_indicies:
                client = self.clients_dict[cid]
                client.set_params(self.server_model_state_dict, self.exclude_layer_keys)

            # clients perform local training
            train_start = time.time()
            client_uploads = []
            for cid in self.active_clients_indicies:
                client = self.clients_dict[cid]
                # client.training(r, client.client_config['num_epochs'])
                client.training(r, kwargs['num_epochs'])
                client_uploads.append(client.upload())
            train_time = time.time() - train_start
            print(f" Training time:{train_time:.3f} seconds")
            # collect training stats
            # average train loss and acc over active clients, where each client uses the latest local models
            self.collect_stats(stage="train", round=r, active_only=True)

            # ranking at these rounds
            if 1 <= r % 55 <= 10 and ("FedMCS" in self.cs_method or self.cs_method == "FedRCS"):
                self.ranking(client_uploads, r)

            # get new server model
            self.pre_global_model = deepcopy(self.server_side_client.model)
            # agg_start = time.time()
            self.aggregate(client_uploads, round=r)
            # agg_time = time.time() - agg_start
            # print(f" Aggregation time:{agg_time:.3f} seconds")
            # collect testing stats
            if (r - 1) % self.server_config['test_every'] == 0:
                test_start = time.time()
                self.testing(round=r, active_only=True)
                test_time = time.time() - test_start
                print(f" Testing time:{test_time:.3f} seconds")
                self.collect_stats(stage="test", round=r, active_only=True)
                print(" avg_test_acc:", self.gfl_test_acc_dict[r]['acc_by_criteria'])
                # print(" pfl_avg_test_acc:", self.average_pfl_test_acc_dict[r])

                if kwargs["metric"] == "bacc":
                    current_gfl_acc = self.gfl_test_acc_dict[r]['acc_by_criteria']['BACC']
                else:
                    current_gfl_acc = self.gfl_test_acc_dict[r]['acc_by_criteria']['uniform']
                if current_gfl_acc > best_test_acc:
                    best_test_acc = current_gfl_acc
                    self.server_model_state_dict_best_so_far = deepcopy(self.server_model_state_dict)
                    tqdm.write(
                        f" Best test accuracy:{float(best_test_acc):5.3f}. Best server model is updatded and saved at {kwargs['filename']}!")
                    if ('filename' in kwargs) and kwargs["save_global"]:
                        torch.save(self.server_model_state_dict_best_so_far, kwargs['filename'])

            # wandb monitoring
            if kwargs['use_wandb']:
                stats = {"avg_train_loss": self.average_train_loss_dict[r],
                         "avg_train_acc": self.average_train_acc_dict[r],
                         "gfl_test_acc_uniform": self.gfl_test_acc_dict[r]['acc_by_criteria']['uniform'],
                         "gfl_test_acc_BACC": self.gfl_test_acc_dict[r]['acc_by_criteria']['BACC']
                         }

                #####for criteria in self.average_pfl_test_acc_dict[r].keys():
                ######stats[f'pfl_test_acc_{criteria}'] = self.average_pfl_test_acc_dict[r][criteria]

                wandb.log(stats)

    def cluster1_setup(self, clients_dict):
        D = []
        for i in range(len(clients_dict)):
            client = clients_dict[i]
            Di = np.sum(client.count_by_class_full.cpu().numpy()).astype(int)
            D.append(Di)

        self.n_cluster = int(self.server_config["num_clients"] * self.server_config["participate_ratio"])  # 10
        epsilon = (10 ** 10)
        D = np.array(D)

        weights = D / np.sum(D)
        augmented_weights = np.array([w * self.n_cluster * epsilon for w in weights])
        ordered_client_idx = np.flip(np.argsort(augmented_weights))

        distri_clusters = np.zeros((self.n_cluster, self.server_config["num_clients"]))  # shape (10, 100)
        k = 0
        for client_idx in ordered_client_idx:
            while augmented_weights[client_idx] > 0 and k < self.n_cluster:
                sum_proba_in_k = np.sum(distri_clusters[k])
                u_i = min(epsilon - sum_proba_in_k, augmented_weights[client_idx])
                distri_clusters[k, client_idx] = u_i
                augmented_weights[client_idx] += -u_i
                sum_proba_in_k = np.sum(distri_clusters[k])
                if sum_proba_in_k == 1 * epsilon:
                    k += 1

        distri_clusters = distri_clusters.astype(float)
        for l in range(self.n_cluster):
            distri_clusters[l] /= np.sum(distri_clusters[l])

        self.distri_clusters = distri_clusters
        self.weights = weights

    def cluster2_cs(self):
        # Get all local gradients
        local_model_params = []
        for idx in self.clients_dict.keys():
            param = [tens.detach().to("cpu").flatten() for tens in list(self.clients_dict[idx].model.parameters())]
            local_model_params += [np.concatenate(param)]
        global_param = [tens.detach().to("cpu").flatten() for tens in list(self.server_side_client.model.parameters())]
        global_param = np.concatenate(global_param)
        print(global_param)
        local_model_grads = [local_param - global_param for local_param in local_model_params]

        # get_matrix_similarity_from_grads
        n_clients = len(local_model_grads)
        sim_matrix = np.zeros((n_clients, n_clients))
        for i, j in product(range(n_clients), range(n_clients)):
            a, b = local_model_grads[i], local_model_grads[j]
            sim_matrix[i, j] = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-5)    # cosine similarity

        # GET THE DENDROGRAM TREE ASSOCIATED
        linkage_matrix = linkage(sim_matrix, "ward")

        # get_clusters_with_alg2
        epsilon = int(10 ** 6)
        # associate each client to a cluster
        link_matrix_p = deepcopy(linkage_matrix)
        augmented_weights = deepcopy(self.weights)
        for i in range(len(link_matrix_p)):
            idx_1, idx_2 = int(link_matrix_p[i, 0]), int(link_matrix_p[i, 1])
            new_weight = np.array(
                [augmented_weights[idx_1] + augmented_weights[idx_2]])
            augmented_weights = np.concatenate((augmented_weights, new_weight))
            link_matrix_p[i, 2] = int(new_weight * epsilon)
        clusters = fcluster(link_matrix_p, int(epsilon / self.n_cluster), criterion="distance")
        n_clients, n_clusters = len(clusters), len(set(clusters))
        print("--- check n_clients, n_clusters:", n_clients, n_clusters)
        # Associate each cluster to its number of clients in the cluster
        pop_clusters = np.zeros((n_clusters, 2)).astype(np.int64)
        for i in range(n_clusters):
            pop_clusters[i, 0] = i + 1
            for client in np.where(clusters == i + 1)[0]:
                pop_clusters[i, 1] += int(self.weights[client] * epsilon * self.n_cluster)
        pop_clusters = pop_clusters[pop_clusters[:, 1].argsort()]
        distri_clusters = np.zeros((self.n_cluster, n_clients)).astype(int)
        # n_sampled biggest clusters that will remain unchanged
        kept_clusters = pop_clusters[n_clusters - self.n_cluster:, 0]
        for idx, cluster in enumerate(kept_clusters):
            for client in np.where(clusters == cluster)[0]:
                distri_clusters[idx, client] = int(self.weights[client] * self.n_cluster * epsilon)
        k = 0
        for j in pop_clusters[: n_clusters - self.n_cluster, 0]:
            clients_in_j = np.where(clusters == j)[0]
            np.random.shuffle(clients_in_j)
            for client in clients_in_j:
                weight_client = int(self.weights[client] * epsilon * self.n_cluster)
                while weight_client > 0:
                    sum_proba_in_k = np.sum(distri_clusters[k])
                    u_i = min(epsilon - sum_proba_in_k, weight_client)
                    distri_clusters[k, client] = u_i
                    weight_client += -u_i
                    sum_proba_in_k = np.sum(distri_clusters[k])
                    if sum_proba_in_k == 1 * epsilon:
                        k += 1
        distri_clusters = distri_clusters.astype(float)
        print(distri_clusters.shape)
        for l in range(self.n_cluster):
            distri_clusters[l] /= np.sum(distri_clusters[l])

        # sample clients
        selected_client_idxs = np.zeros(self.n_cluster, dtype=int)
        all_idx = np.arange(self.server_config["num_clients"])
        for k in range(self.n_cluster):
            selected_client_idxs[k] = int(np.random.choice(all_idx, 1, p=distri_clusters[k]))

        return selected_client_idxs

    def oort_cs(self):
        # Update client utilities
        active_clients_before = self.active_clients_indicies
        for idx in active_clients_before:
            self.client_last_rounds[idx] = self.current_round - 1       # cuz select in previous round
            statistical_utility = np.sqrt(self.clients_dict[idx].D_client * self.clients_dict[idx].loss_utility)
            client_utility = statistical_utility + np.sqrt(
                0.1 * np.log(self.current_round - 1) / self.client_last_rounds[idx])    # cuz select in previous round
            self.client_utilities[idx] = client_utility

        """client selection here"""
        selected_clients = []

        if self.current_round > 1:      # check how about = 1?
            # Exploitation
            exploited_clients_count = max(math.ceil(0.1 * self.n_selected), self.n_selected - len(self.unexplored_clients))
            sorted_by_utility = sorted(self.client_utilities, key=self.client_utilities.get, reverse=True)

            # Calculate cut-off utility
            cut_off_util = (self.client_utilities[sorted_by_utility[exploited_clients_count - 1]] * self.cut_off)

            # Include clients with utilities higher than the cut-off
            exploited_clients = []
            for idx in sorted_by_utility:
                if self.client_utilities[idx] > cut_off_util and idx not in self.blacklist:
                    exploited_clients.append(idx)

            # Sample clients with their utilities
            total_utility = float(sum(self.client_utilities[idx] for idx in exploited_clients))
            probabilities = [self.client_utilities[idx] / total_utility for idx in exploited_clients]
            if len(probabilities) > 0 and exploited_clients_count > 0:
                selected_clients = np.random.choice(exploited_clients, min(len(exploited_clients), exploited_clients_count),
                                                        p=probabilities, replace=False)
                selected_clients = selected_clients.tolist()

            # If the result of exploitation wasn't enough to meet the required length
            if len(selected_clients) < exploited_clients_count:
                last_index = sorted_by_utility.index(exploited_clients[-1]) if exploited_clients else 0
                for idx in range(last_index + 1, len(sorted_by_utility)):
                    if not sorted_by_utility[idx] in self.blacklist and len(selected_clients) < exploited_clients_count:
                        selected_clients.append(sorted_by_utility[idx])

        # Exploration - Select unexplored clients randomly
        selected_unexplore_clients = np.random.choice(self.unexplored_clients, self.n_selected - len(selected_clients),
                                                      replace=False).tolist()
        print("check selected old and new:", selected_clients, selected_unexplore_clients)
        print("check blacklist:", self.blacklist)
        selected_clients += selected_unexplore_clients

        # Update after select
        for idx in selected_unexplore_clients:
            self.unexplored_clients.remove(idx)
        for idx in selected_clients:
            self.client_selected_times[idx] += 1
            if self.client_selected_times[idx] > self.blacklist_num:
                self.blacklist.append(idx)

        return selected_clients

    def oort2_cs(self):
        # Update client utilities
        active_clients_before = self.active_clients_indicies
        for idx in active_clients_before:
            self.client_last_rounds[idx] = self.current_round - 1       # cuz select in previous round
            statistical_utility = np.sqrt(self.clients_dict[idx].D_client * self.clients_dict[idx].loss_utility)
            self.client_utilities[idx] = statistical_utility + np.sqrt(
                0.1 * np.log(self.current_round - 1) / (self.current_round - 1))    # cuz select in previous round

        active_clients_past = list(set(range(0, self.n_all_clients)) - set(active_clients_before) - set(self.unexplored_clients))
        for idx in active_clients_past:
            self.client_utilities[idx] = (self.client_utilities[idx] -
                                          np.sqrt(0.1 * np.log(self.current_round - 2) / self.client_last_rounds[idx]) +
                                          np.sqrt(0.1 * np.log(self.current_round - 1) / self.client_last_rounds[idx]))

        """client selection here"""
        selected_clients = []

        if self.current_round > 1:      # check how about = 1?
            # Exploitation
            exploited_clients_count = max(math.ceil(0.1 * self.n_selected), self.n_selected - len(self.unexplored_clients))
            sorted_by_utility = sorted(self.client_utilities, key=self.client_utilities.get, reverse=True)

            # Calculate cut-off utility
            cut_off_util = (self.client_utilities[sorted_by_utility[exploited_clients_count - 1]] * self.cut_off)

            # Include clients with utilities higher than the cut-off
            exploited_clients = []
            for idx in sorted_by_utility:
                if self.client_utilities[idx] > cut_off_util and idx not in self.blacklist:
                    exploited_clients.append(idx)

            # Sample clients with their utilities
            total_utility = float(sum(self.client_utilities[idx] for idx in exploited_clients))
            probabilities = [self.client_utilities[idx] / total_utility for idx in exploited_clients]
            if len(probabilities) > 0 and exploited_clients_count > 0:
                selected_clients = np.random.choice(exploited_clients, min(len(exploited_clients), exploited_clients_count),
                                                        p=probabilities, replace=False)
                selected_clients = selected_clients.tolist()

            # If the result of exploitation wasn't enough to meet the required length
            if len(selected_clients) < exploited_clients_count:
                last_index = sorted_by_utility.index(exploited_clients[-1]) if exploited_clients else 0
                for idx in range(last_index + 1, len(sorted_by_utility)):
                    if not sorted_by_utility[idx] in self.blacklist and len(selected_clients) < exploited_clients_count:
                        selected_clients.append(sorted_by_utility[idx])

        # Exploration - Select unexplored clients randomly
        selected_unexplore_clients = np.random.choice(self.unexplored_clients, self.n_selected - len(selected_clients),
                                                      replace=False).tolist()
        print("check selected old and new:", selected_clients, selected_unexplore_clients)
        print("check blacklist:", self.blacklist)
        selected_clients += selected_unexplore_clients

        # Update after select
        for idx in selected_unexplore_clients:
            self.unexplored_clients.remove(idx)
        for idx in selected_clients:
            self.client_selected_times[idx] += 1
            if self.client_selected_times[idx] > self.blacklist_num:
                self.blacklist.append(idx)

        return selected_clients

    def tqm_cs(self):
        print("---client_selected_times:", self.client_selected_times)
        # Update client utilities
        active_clients_before = self.active_clients_indicies
        count_per_class_test = Counter(self.server_side_client.testloader.dataset.targets.numpy())
        num_classes = self.server_side_client.client_config['num_classes']
        count_per_class_test = torch.tensor([count_per_class_test[cls] for cls in range(num_classes)])
        for idx in active_clients_before:
            self.clients_dict[idx].testing(self.current_round, testloader=self.server_side_client.testloader)  # for Cifar10 exps, if use proxy set: testloader=self.validloader
            acc_per_class = self.clients_dict[idx].test_acc_dict[self.current_round]['correct_per_class']
            # mul_acc = torch.prod(np.exp(acc_per_class / count_per_class_test)).detach().numpy()
            # mul_acc = torch.prod(1 + (acc_per_class / count_per_class_test)).detach().numpy()
            mul_acc = np.prod(1 + (acc_per_class / count_per_class_test).detach().numpy()) ** (1/num_classes) - 1

            self.client_last_rounds[idx] = self.current_round - 1       # cuz select in previous round
            statistical_utility = self.clients_dict[idx].D_client * mul_acc
            # self.client_utilities[idx] = statistical_utility / np.log(self.client_selected_times[idx]+1)
            self.client_utilities[idx] = statistical_utility

        active_clients_past = list(set(range(0, self.n_all_clients)) - set(active_clients_before) - set(self.unexplored_clients))
        for idx in active_clients_past:
            # re_hat_of_e = (np.sqrt((self.current_round-1)/self.client_last_rounds[idx] - 1) -
            #                np.sqrt((self.current_round-2)/self.client_last_rounds[idx] - 1))
            # self.client_utilities[idx] *= np.exp(re_hat_of_e)
            # self.client_utilities[idx] *= (self.current_round-1) / (self.current_round-2)
            self.client_utilities[idx] *= (np.sqrt(self.current_round - self.client_last_rounds[idx]) /
                                           np.sqrt((self.current_round-1) - self.client_last_rounds[idx]))

        print("---client_utilities:", self.client_utilities)

        """client selection here"""
        selected_clients = []

        if self.current_round > 1:      # check how about = 1?
            # Exploitation
            exploited_clients_count = max(math.ceil(0.1 * self.n_selected), self.n_selected - len(self.unexplored_clients))
            sorted_by_utility = sorted(self.client_utilities, key=self.client_utilities.get, reverse=True)

            # Calculate cut-off utility
            cut_off_util = (self.client_utilities[sorted_by_utility[exploited_clients_count - 1]] * self.cut_off)

            # Include clients with utilities higher than the cut-off
            exploited_clients = []
            for idx in sorted_by_utility:
                if self.client_utilities[idx] > cut_off_util and idx not in self.blacklist:
                    exploited_clients.append(idx)

            # Sample clients with their utilities
            total_utility = float(sum(self.client_utilities[idx] for idx in exploited_clients))
            probabilities = [self.client_utilities[idx] / total_utility for idx in exploited_clients]
            if len(probabilities) > 0 and exploited_clients_count > 0:
                selected_clients = np.random.choice(exploited_clients, min(len(exploited_clients), exploited_clients_count),
                                                        p=probabilities, replace=False)
                selected_clients = selected_clients.tolist()

            # If the result of exploitation wasn't enough to meet the required length
            if len(selected_clients) < exploited_clients_count:
                last_index = sorted_by_utility.index(exploited_clients[-1]) if exploited_clients else 0
                for idx in range(last_index + 1, len(sorted_by_utility)):
                    if not sorted_by_utility[idx] in self.blacklist and len(selected_clients) < exploited_clients_count:
                        selected_clients.append(sorted_by_utility[idx])

        # Exploration - Select unexplored clients randomly
        selected_unexplore_clients = np.random.choice(self.unexplored_clients, self.n_selected - len(selected_clients),
                                                      replace=False).tolist()
        print("check selected old and new:", selected_clients, selected_unexplore_clients)
        print("check blacklist:", self.blacklist)
        selected_clients += selected_unexplore_clients

        # Update after select
        for idx in selected_unexplore_clients:
            self.unexplored_clients.remove(idx)
        for idx in selected_clients:
            self.client_selected_times[idx] += 1
            if self.client_selected_times[idx] > self.blacklist_num:
                self.blacklist.append(idx)

        return selected_clients

    def tqm2_cs(self):
        print("---client_selected_times:", self.client_selected_times)
        # Update client utilities
        active_clients_before = self.active_clients_indicies

        global_param = [tens.detach().to("cpu").flatten() for tens in list(self.server_side_client.model.parameters())]
        global_param = np.concatenate(global_param)
        global_param_before = [tens.detach().to("cpu").flatten() for tens in list(self.pre_global_model.parameters())]
        global_param_before = np.concatenate(global_param_before)
        global_gradient = global_param - global_param_before
        for idx in active_clients_before:
            param = [tens.detach().to("cpu").flatten() for tens in list(self.clients_dict[idx].model.parameters())]
            param = np.concatenate(param)
            gradient = param - global_param_before
            cos_sim = np.dot(gradient, global_gradient) / (np.linalg.norm(gradient) * np.linalg.norm(global_gradient))

            self.client_last_rounds[idx] = self.current_round - 1  # cuz select in previous round
            statistical_utility = self.clients_dict[idx].D_client * max(cos_sim, 0.1)
            # self.client_utilities[idx] = statistical_utility / np.log(self.client_selected_times[idx] + 1)
            self.client_utilities[idx] = statistical_utility
            print("---cos_sim, D_client:", cos_sim, self.clients_dict[idx].D_client)

        active_clients_past = list(set(range(0, self.n_all_clients)) - set(active_clients_before) - set(self.unexplored_clients))
        for idx in active_clients_past:
            re_hat_of_e = (np.sqrt((self.current_round - 1) / self.client_last_rounds[idx] - 1) -
                           np.sqrt((self.current_round - 2) / self.client_last_rounds[idx] - 1))
            self.client_utilities[idx] *= np.exp(re_hat_of_e)
        print("---client_utilities:", self.client_utilities)

        """client selection here"""
        selected_clients = []

        if self.current_round > 1:  # check how about = 1?
            # Exploitation
            exploited_clients_count = max(math.ceil(0.1 * self.n_selected),
                                          self.n_selected - len(self.unexplored_clients))
            sorted_by_utility = sorted(self.client_utilities, key=self.client_utilities.get, reverse=True)

            # Calculate cut-off utility
            cut_off_util = (self.client_utilities[sorted_by_utility[exploited_clients_count - 1]] * self.cut_off)

            # Include clients with utilities higher than the cut-off
            exploited_clients = []
            for idx in sorted_by_utility:
                if self.client_utilities[idx] > cut_off_util and idx not in self.blacklist:
                    exploited_clients.append(idx)

            # Sample clients with their utilities
            total_utility = float(sum(self.client_utilities[idx] for idx in exploited_clients))
            probabilities = [self.client_utilities[idx] / total_utility for idx in exploited_clients]
            if len(probabilities) > 0 and exploited_clients_count > 0:
                selected_clients = np.random.choice(exploited_clients,
                                                    min(len(exploited_clients), exploited_clients_count),
                                                    p=probabilities, replace=False)
                selected_clients = selected_clients.tolist()

            # If the result of exploitation wasn't enough to meet the required length
            if len(selected_clients) < exploited_clients_count:
                last_index = sorted_by_utility.index(exploited_clients[-1]) if exploited_clients else 0
                for idx in range(last_index + 1, len(sorted_by_utility)):
                    if not sorted_by_utility[idx] in self.blacklist and len(selected_clients) < exploited_clients_count:
                        selected_clients.append(sorted_by_utility[idx])

        # Exploration - Select unexplored clients randomly
        selected_unexplore_clients = np.random.choice(self.unexplored_clients, self.n_selected - len(selected_clients),
                                                      replace=False).tolist()
        print("check selected old and new:", selected_clients, selected_unexplore_clients)
        print("check blacklist:", self.blacklist)
        selected_clients += selected_unexplore_clients

        # Update after select
        for idx in selected_unexplore_clients:
            self.unexplored_clients.remove(idx)
        for idx in selected_clients:
            self.client_selected_times[idx] += 1
            if self.client_selected_times[idx] > self.blacklist_num:
                self.blacklist.append(idx)

        return selected_clients

    def moort_cs(self):
        print("---client_selected_times:", self.client_selected_times)
        # Update client utilities
        active_clients_before = self.active_clients_indicies
        count_per_class_test = Counter(self.server_side_client.testloader.dataset.targets.numpy())
        num_classes = self.server_side_client.client_config['num_classes']
        count_per_class_test = torch.tensor([count_per_class_test[cls] for cls in range(num_classes)])

        global_param = [tens.detach().to("cpu").flatten() for tens in list(self.server_side_client.model.parameters())]
        global_param = np.concatenate(global_param)
        global_param_before = [tens.detach().to("cpu").flatten() for tens in list(self.pre_global_model.parameters())]
        global_param_before = np.concatenate(global_param_before)
        global_gradient = global_param - global_param_before

        for idx in active_clients_before:
            self.clients_dict[idx].testing(self.current_round, testloader=self.server_side_client.testloader)  # for Cifar10 exps, if use proxy set: testloader=self.validloader
            acc_per_class = self.clients_dict[idx].test_acc_dict[self.current_round]['correct_per_class']
            # mul_acc = torch.prod(np.exp(acc_per_class / count_per_class_test)).detach().numpy()
            # mul_acc = torch.prod(1 + (acc_per_class / count_per_class_test)).detach().numpy()
            mul_acc = np.prod(1 + (acc_per_class / count_per_class_test).detach().numpy()) ** (1/num_classes) - 1

            param = [tens.detach().to("cpu").flatten() for tens in list(self.clients_dict[idx].model.parameters())]
            param = np.concatenate(param)
            gradient = param - global_param_before
            cos_sim = np.dot(gradient, global_gradient) / (np.linalg.norm(gradient) * np.linalg.norm(global_gradient))

            self.client_last_rounds[idx] = self.current_round - 1       # cuz select in previous round
            statistical_utility = self.clients_dict[idx].D_client * mul_acc * max(cos_sim, 0.1)
            # self.client_utilities[idx] = statistical_utility / np.log(self.client_selected_times[idx]+1)
            self.client_utilities[idx] = statistical_utility

        active_clients_past = list(set(range(0, self.n_all_clients)) - set(active_clients_before) - set(self.unexplored_clients))
        for idx in active_clients_past:
            # re_hat_of_e = (np.sqrt((self.current_round-1)/self.client_last_rounds[idx] - 1) -
            #                np.sqrt((self.current_round-2)/self.client_last_rounds[idx] - 1))
            # self.client_utilities[idx] *= np.exp(re_hat_of_e)
            # self.client_utilities[idx] *= (self.current_round-1) / (self.current_round-2)
            self.client_utilities[idx] *= (np.sqrt(self.current_round - self.client_last_rounds[idx]) /
                                           np.sqrt((self.current_round-1) - self.client_last_rounds[idx]))

        print("---client_utilities:", self.client_utilities)

        """client selection here"""
        selected_clients = []

        if self.current_round > 1:      # check how about = 1?
            # Exploitation
            exploited_clients_count = max(math.ceil(0.1 * self.n_selected), self.n_selected - len(self.unexplored_clients))
            sorted_by_utility = sorted(self.client_utilities, key=self.client_utilities.get, reverse=True)

            # Calculate cut-off utility
            cut_off_util = (self.client_utilities[sorted_by_utility[exploited_clients_count - 1]] * self.cut_off)

            # Include clients with utilities higher than the cut-off
            exploited_clients = []
            for idx in sorted_by_utility:
                if self.client_utilities[idx] > cut_off_util and idx not in self.blacklist:
                    exploited_clients.append(idx)

            # Sample clients with their utilities
            total_utility = float(sum(self.client_utilities[idx] for idx in exploited_clients))
            probabilities = [self.client_utilities[idx] / total_utility for idx in exploited_clients]
            if len(probabilities) > 0 and exploited_clients_count > 0:
                selected_clients = np.random.choice(exploited_clients, min(len(exploited_clients), exploited_clients_count),
                                                        p=probabilities, replace=False)
                selected_clients = selected_clients.tolist()

            # If the result of exploitation wasn't enough to meet the required length
            if len(selected_clients) < exploited_clients_count:
                last_index = sorted_by_utility.index(exploited_clients[-1]) if exploited_clients else 0
                for idx in range(last_index + 1, len(sorted_by_utility)):
                    if not sorted_by_utility[idx] in self.blacklist and len(selected_clients) < exploited_clients_count:
                        selected_clients.append(sorted_by_utility[idx])

        # Exploration - Select unexplored clients randomly
        selected_unexplore_clients = np.random.choice(self.unexplored_clients, self.n_selected - len(selected_clients),
                                                      replace=False).tolist()
        print("check selected old and new:", selected_clients, selected_unexplore_clients)
        print("check blacklist:", self.blacklist)
        selected_clients += selected_unexplore_clients

        # Update after select
        for idx in selected_unexplore_clients:
            self.unexplored_clients.remove(idx)
        for idx in selected_clients:
            self.client_selected_times[idx] += 1
            if self.client_selected_times[idx] > self.blacklist_num:
                self.blacklist.append(idx)

        return selected_clients




