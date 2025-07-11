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


class FedAvgClient(Client):
    def __init__(self, criterion, trainset, testset, client_config, cid, device, **kwargs):
        super().__init__(criterion, trainset, testset,
                         client_config, cid, device, **kwargs)
        self._initialize_model()

        temp = [self.count_by_class[cls] if cls in self.count_by_class.keys() else 1e-12 for cls in
                range(client_config['num_classes'])]
        self.count_by_class_full = torch.tensor(temp).to(self.device)
        # print(np.array(temp).astype(int).tolist(), ",")  # dem so mau moi class trong moi client
        # print("*** CHECK DATA *** : ", np.sum(np.array(temp).astype(int)))  # dem so mau moi client

        self.p = np.array(temp) / np.sum(np.array(temp))
        self.entropy = - np.sum(self.p * (np.log(self.p)))
        self.score = np.sum(np.array(temp)) * self.entropy
        # print(self.entropy)
        print(self.score)

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
        for i in range(num_epochs):
            epoch_loss, correct = 0.0, 0
            for j, (x, y) in enumerate(self.trainloader):
<<<<<<< HEAD
                # if len(y) == 1:
                #     continue
=======
                if len(y) == 1:
                    continue
>>>>>>> ebb23b9832dc61af2d903c511c954096fcb5628d
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
            epoch_loss /= len(self.trainloader.dataset)
            epoch_accuracy = correct / len(self.trainloader.dataset)
            loss_seq.append(epoch_loss)
            acc_seq.append(epoch_accuracy)
        self.new_state_dict = self.model.state_dict()
        self.train_loss_dict[round] = loss_seq
        self.train_acc_dict[round] = acc_seq

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
        with torch.no_grad():
            for i, (x, y) in enumerate(testloader):
                # forward pass
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model.forward(x)
                # stats
                predicted = yhat.data.max(1)[1]
                #######classes_shown_in_this_batch = torch.unique(y).cpu().numpy()
                #######for cls in classes_shown_in_this_batch:
                for cls in range(num_classes):
                    test_correct_per_class[cls] += ((predicted == y) * (y == cls)).sum().item()
                    test_TN_per_class[cls] += ((predicted != cls) * (y != cls)).sum().item()
        acc_by_critertia_dict = {}
        for k in weight_per_class_dict.keys():
            acc_by_critertia_dict[k] = (((weight_per_class_dict[k] * test_correct_per_class).sum()) /
                                        ((weight_per_class_dict[k] * test_count_per_class).sum())).item()

        acc_by_critertia_dict["BACC"] = ((test_correct_per_class / test_count_per_class +
                                          test_TN_per_class / (
                                                  test_count_per_class.sum() - test_count_per_class)) / 2).mean()

        self.test_acc_dict[round] = {'acc_by_criteria': acc_by_critertia_dict,
                                     'correct_per_class': test_correct_per_class,
                                     'weight_per_class': weight_per_class_dict}


class FedAvgServer(Server):
    def __init__(self, server_config, clients_dict, exclude, cs_method, **kwargs):
        super().__init__(server_config, clients_dict, **kwargs)
        self.summary_setup()
        self.server_model_state_dict = deepcopy(self.clients_dict[0].get_params())
        # make sure the starting point is correct
        self.server_side_client.set_params(self.server_model_state_dict, exclude_keys=set())
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

        self.sort_client_entropy = np.zeros(shape=(10, 10), dtype=int)
<<<<<<< HEAD
        self.sort_client_ref = np.arange(100,).reshape((10, 10))
=======
        self.sort_client_ref = np.zeros(shape=(10, 10), dtype=int)
>>>>>>> ebb23b9832dc61af2d903c511c954096fcb5628d
        self.sort_client = np.arange(100,).reshape((10, 10))

        self.cluster1_setup(clients_dict)

<<<<<<< HEAD
        self.validloader = DataLoader(kwargs["global_validset"], batch_size=128, shuffle=False) if kwargs["global_validset"] else None

=======
>>>>>>> ebb23b9832dc61af2d903c511c954096fcb5628d
    def ranking(self, client_uploads, round):
        res = round % 55
        assert 1 <= res <= 10, "false round"

<<<<<<< HEAD
        D_list = []
=======
>>>>>>> ebb23b9832dc61af2d903c511c954096fcb5628d
        mul_acc_list = []
        entropy_list = []
        ref_list = []
        for idx, (client_state_dict, count_per_class_client, ref_score) in enumerate(client_uploads):
<<<<<<< HEAD
=======
            self.server_side_client.set_params(client_state_dict, self.exclude_layer_keys)
            self.server_side_client.testing(round, testloader=None)  # use global testdataset
>>>>>>> ebb23b9832dc61af2d903c511c954096fcb5628d
            count_per_class_test = Counter(self.server_side_client.testloader.dataset.targets.numpy())
            num_classes = self.server_side_client.client_config['num_classes']
            count_per_class_test = torch.tensor([count_per_class_test[cls] * 1.0 for cls in range(num_classes)])
            # print("---check count_by_class_testset : ", count_per_class_test)
<<<<<<< HEAD
            # print("---check count_by_class_client : ", count_per_class_client)
            num_data_in_client = torch.sum(count_per_class_client)
            # print("---check num_data_in_client : ", num_data_in_client)
            D_list.append(num_data_in_client)

            self.server_side_client.set_params(client_state_dict, self.exclude_layer_keys)
            self.server_side_client.testing(round, testloader=self.validloader)  # use global testdataset
            self.gfl_test_acc_dict[round] = self.server_side_client.test_acc_dict[round]
            acc_per_class = self.gfl_test_acc_dict[round]['correct_per_class']
            # print("---check acc_per_class : ", acc_per_class)
            mul_acc = torch.prod(np.exp(acc_per_class / count_per_class_test))
            mul_acc_list.append(mul_acc.cpu().numpy())
=======
            print("---check count_by_class_client : ", count_per_class_client)
            num_data_in_client = torch.sum(count_per_class_client)
            print("---check num_data_in_client : ", num_data_in_client)
            self.gfl_test_acc_dict[round] = self.server_side_client.test_acc_dict[round]
            acc_per_class = self.gfl_test_acc_dict[round]['correct_per_class']
            print("---check acc_per_class : ", acc_per_class)
>>>>>>> ebb23b9832dc61af2d903c511c954096fcb5628d

            entropy_list.append(ref_score[0])
            ref_list.append(ref_score[1])

<<<<<<< HEAD
        # Calculate gradient list
        gradients_list = []
=======
            mul_acc = torch.prod(np.exp(acc_per_class / count_per_class_test)) * num_data_in_client
            mul_acc_list.append(mul_acc.cpu().numpy())

        print("---check mul_acc_list : ", mul_acc_list)

        sorted_idx_acc = [x for _, x in sorted(zip(mul_acc_list, self.active_clients_indicies), reverse=True)]
        sorted_idx_acc = np.array(sorted_idx_acc)
        print("---check sorted_idx_acc : ", sorted_idx_acc)

        # Calculate gradient list
        gradients_list = []
        # num_samples_list = []
>>>>>>> ebb23b9832dc61af2d903c511c954096fcb5628d
        for idx, (client_state_dict, count_per_class_client, _) in enumerate(client_uploads):
            gradient = []
            for state_key in client_state_dict.keys():
                if state_key not in self.exclude_layer_keys:
                    g = client_state_dict[state_key] - self.server_model_state_dict[state_key]
                    gradient.append(g.cpu().numpy().flatten())
            gradient = np.concatenate(gradient, axis=0)
            gradients_list.append(gradient)
<<<<<<< HEAD
=======
            # num_samples_list.append(torch.sum(count_per_class_client))
>>>>>>> ebb23b9832dc61af2d903c511c954096fcb5628d

        weights = torch.tensor([1] * 10)
        # weights = torch.tensor(num_samples_list)
        weights = weights / torch.sum(weights)
        weights_2d = weights.unsqueeze(1)
        gradients_list = torch.tensor(gradients_list)
        gradient_global = gradients_list * weights_2d
        gradient_global = torch.sum(gradient_global, 0)
        cosin_similary_list = [torch.nn.CosineSimilarity(dim=0)(gradient_global, grad_k) for grad_k in
                               gradients_list]
<<<<<<< HEAD

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
        else:   # FedMCS5
            score_final = [D_list[i] * mul_acc_list[i] * cosin_similary_list[i] for i in range(len(mul_acc_list))]

        sorted_idx = np.array([x for _, x in sorted(zip(score_final, self.active_clients_indicies), reverse=True)])
        print("---check sorted_idx_score : ", sorted_idx)
        self.sort_client[res - 1] = sorted_idx

        print("---check ref_list : ", ref_list)
        print("---check active_clients_indicies : ", self.active_clients_indicies)

=======
        print("---check cosin_similary_list : ", cosin_similary_list)
        sorted_idx_grad = np.array(
            [x for _, x in sorted(zip(cosin_similary_list, self.active_clients_indicies), reverse=True)])
        print("---check sorted_idx_grad : ", sorted_idx_grad)

        # Sort
        mul_acc_list = torch.tensor(np.array(mul_acc_list))
        score_final = [mul_acc_list[i] * cosin_similary_list[i] for i in range(len(mul_acc_list))]
        sorted_idx = np.array(
            [x for _, x in sorted(zip(score_final, self.active_clients_indicies), reverse=True)])
        print("---check sorted_idx_score : ", sorted_idx)
        self.sort_client[res - 1] = sorted_idx

        print("---check entropy_list : ", entropy_list)
        sorted_entropy = np.array(
            [x for _, x in sorted(zip(entropy_list, self.active_clients_indicies), reverse=True)])
        print("---check sorted_entropy : ", sorted_entropy)
        self.sort_client_entropy[res - 1] = sorted_entropy

        print("---check ref_list : ", ref_list)
>>>>>>> ebb23b9832dc61af2d903c511c954096fcb5628d
        sorted_ref = np.array([x for _, x in sorted(zip(ref_list, self.active_clients_indicies), reverse=True)])
        print("---check sorted_ref : ", sorted_ref)
        self.sort_client_ref[res - 1] = sorted_ref

        if res == 10:
            self.sort_client = self.sort_client.transpose()
            print("---check self.sort_client : ")
            print(self.sort_client)

<<<<<<< HEAD
=======
            self.sort_client_entropy = self.sort_client_entropy.transpose()
            print("---check self.sort_client_entropy : ")
            print(self.sort_client_entropy)

>>>>>>> ebb23b9832dc61af2d903c511c954096fcb5628d
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
        # # test the performance for local models (potentiallt only for active local clients)
        # client_indices = self.clients_dict.keys()
        # if active_only:
        #     client_indices = self.active_clients_indicies
        # for cid in client_indices:
        #     client = self.clients_dict[cid]
        #     # test local model on the splitted testset
        #     if self.server_config['split_testset'] == True:
        #         client.testing(round, None)
        #     else:
        #         # test local model on the global testset
        #         client.testing(round, self.server_side_client.testloader)

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

            if self.cs_method == "Random":
                self.active_clients_indicies = self.select_clients(self.server_config['participate_ratio'])

            elif self.cs_method == "FedMCS":
                if r <= 10:
                    self.active_clients_indicies = np.arange((r - 1) * 10, r * 10)
                elif r <= 10 + kwargs["T21"]:
                    self.active_clients_indicies = self.sort_client[0]
                elif 10 + kwargs["T21"] < r <= 10 + kwargs["T21"] + 15 * kwargs["l"]:
                    rr = (r - 11 - kwargs["T21"]) % 15
                    k = int(np.abs(np.sqrt(2 * (rr + 1) + 1 / 4) - 1 / 2))
                    residual = int((rr + 1) - k * (k + 1) / 2)
                    if residual == 1:
                        clients_consider = self.sort_client[:k + 1].flatten()
                        np.random.shuffle(clients_consider)
                        print("round :", r, "k :", k, "res :", residual)
                        print("client consider :", clients_consider)
                    elif residual == 0 and k == 1:
                        clients_consider = self.sort_client[0]
                        print("client consider :", clients_consider)
                    self.active_clients_indicies = clients_consider[10 * (
                            residual - 1):10 * residual] if residual > 0 else clients_consider[-10:]
                else:
                    self.active_clients_indicies = self.select_clients(self.server_config['participate_ratio'])

<<<<<<< HEAD
            elif self.cs_method in ["FedMCS5", "FedMCS*", "FedMCSda", "FedMCSdg", "FedMCSag"]:
=======
            elif self.cs_method == "FedMCS5":
>>>>>>> ebb23b9832dc61af2d903c511c954096fcb5628d
                if r <= 10:
                    self.active_clients_indicies = self.sort_client[r-1]
                else:
                    rr = (r-11) % 55
                    k = int(np.abs(np.sqrt(2 * (rr + 1) + 1 / 4) - 1 / 2))
                    residual = int((rr + 1) - k * (k + 1) / 2)
                    if residual == 1:
                        clients_consider = self.sort_client[:k + 1].flatten()
                        np.random.shuffle(clients_consider)
                        print("round :", r, "k :", k, "res :", residual)
                        print("client consider :", clients_consider)
                    elif residual == 0 and k == 1:
                        clients_consider = self.sort_client[0]
                        print("client consider :", clients_consider)
                    self.active_clients_indicies = clients_consider[10 * (
                            residual - 1):10 * residual] if residual > 0 else clients_consider[-10:]

            elif self.cs_method == "Cluster1":
                self.active_clients_indicies = np.zeros(shape=(self.n_cluster,))
                all_idx = np.arange(self.server_config["num_clients"])  # (0, 1, ..., 99)
                for k in range(self.n_cluster):
                    weights = self.distri_clusters[k]
                    self.active_clients_indicies[k] = int(np.random.choice(all_idx, 1, p=weights / sum(weights)))

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
            if 1 <= r % 55 <= 10 and "FedMCS" in self.cs_method:
                self.ranking(client_uploads, r)

            # get new server model
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
