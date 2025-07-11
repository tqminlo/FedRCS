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
from .FedUH import FedUHClient, FedUHServer
import math
import time
import torch.nn.functional as F


class FedNHClient(FedUHClient):
    def __init__(self, criterion, trainset, testset, client_config, cid, device, **kwargs):
        super().__init__(criterion, trainset, testset, client_config, cid, device, **kwargs)

    def _estimate_prototype(self):
        self.model.eval()
        self.model.return_embedding = True
        embedding_dim = self.model.prototype.shape[1]
        prototype = torch.zeros_like(self.model.prototype)
        with torch.no_grad():
            for i, (x, y) in enumerate(self.trainloader):
                # forward pass
                x, y = x.to(self.device), y.to(self.device)
                # feature_embedding is normalized
                feature_embedding, _ = self.model.forward(x)
                classes_shown_in_this_batch = torch.unique(y).cpu().numpy()
                for cls in classes_shown_in_this_batch:
                    mask = (y == cls)
                    feature_embedding_in_cls = torch.sum(feature_embedding[mask, :], dim=0)
                    prototype[cls] += feature_embedding_in_cls
        for cls in self.count_by_class.keys():
            # sample mean
            prototype[cls] /= self.count_by_class[cls]
            # normalization so that self.W.data is of the sampe scale as prototype_cls_norm
            prototype_cls_norm = torch.norm(prototype[cls]).clamp(min=1e-12)
            prototype[cls] = torch.div(prototype[cls], prototype_cls_norm)

            # reweight it for aggregartion
            prototype[cls] *= self.count_by_class[cls]

        self.model.return_embedding = False

        to_share = {'scaled_prototype': prototype, 'count_by_class_full': self.count_by_class_full}
        return to_share

    def _estimate_prototype_adv(self):
        self.model.eval()
        self.model.return_embedding = True
        embeddings = []
        labels = []
        weights = []
        prototype = torch.zeros_like(self.model.prototype)
        with torch.no_grad():
            for i, (x, y) in enumerate(self.trainloader):
                # forward pass
                x, y = x.to(self.device), y.to(self.device)
                # feature_embedding is normalized
                # use the latest prototype
                feature_embedding, logits = self.model.forward(x)
                prob_ = F.softmax(logits, dim=1)
                prob = torch.gather(prob_, dim=1, index=y.view(-1, 1))
                labels.append(y)
                weights.append(prob)
                embeddings.append(feature_embedding)
        self.model.return_embedding = False
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        weights = torch.cat(weights, dim=0).view(-1, 1)
        for cls in self.count_by_class.keys():
            mask = (labels == cls)
            weights_in_cls = weights[mask, :]
            feature_embedding_in_cls = embeddings[mask, :]
            prototype[cls] = torch.sum(feature_embedding_in_cls * weights_in_cls, dim=0) / torch.sum(weights_in_cls)
            prototype_cls_norm = torch.norm(prototype[cls]).clamp(min=1e-12)
            prototype[cls] = torch.div(prototype[cls], prototype_cls_norm)

        # calculate predictive power
        to_share = {'adv_agg_prototype': prototype, 'count_by_class_full': self.count_by_class_full}
        return to_share

    def upload(self):
        if self.client_config['FedNH_client_adv_prototype_agg']:
            return self.new_state_dict, self._estimate_prototype_adv(), (self.entropy, self.score)
        else:
            return self.new_state_dict, self._estimate_prototype(), (self.entropy, self.score)


class FedNHServer(FedUHServer):
    def __init__(self, server_config, clients_dict, exclude, cs_method, **kwargs):
        super().__init__(server_config, clients_dict, exclude, cs_method, **kwargs)
        if len(self.exclude_layer_keys) > 0:
            print(f"FedNHServer: the following keys will not be aggregated:\n ", self.exclude_layer_keys)

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
        for idx, (client_state_dict, prototype_dict, ref_score) in enumerate(client_uploads):
<<<<<<< HEAD
=======
            self.server_side_client.set_params(client_state_dict, self.exclude_layer_keys)
            self.server_side_client.testing(round, testloader=None)  # use global testdataset
>>>>>>> ebb23b9832dc61af2d903c511c954096fcb5628d
            count_per_class_test = Counter(self.server_side_client.testloader.dataset.targets.numpy())
            num_classes = self.server_side_client.client_config['num_classes']
            count_per_class_test = torch.tensor([count_per_class_test[cls] * 1.0 for cls in range(num_classes)])
            # print("---check count_by_class_testset : ", count_per_class_test)
            count_per_class_client = prototype_dict['count_by_class_full']
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
        for idx, (client_state_dict, prototype_dict, _) in enumerate(client_uploads):
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
        else:  # FedMCS5
            score_final = [D_list[i] * mul_acc_list[i] * cosin_similary_list[i] for i in range(len(mul_acc_list))]

=======
        print("---check cosin_similary_list : ", cosin_similary_list)
        sorted_idx_grad = np.array(
            [x for _, x in sorted(zip(cosin_similary_list, self.active_clients_indicies), reverse=True)])
        print("---check sorted_idx_grad : ", sorted_idx_grad)

        # Sort
        mul_acc_list = torch.tensor(np.array(mul_acc_list))
        score_final = [mul_acc_list[i] * cosin_similary_list[i] for i in range(len(mul_acc_list))]
>>>>>>> ebb23b9832dc61af2d903c511c954096fcb5628d
        sorted_idx = np.array(
            [x for _, x in sorted(zip(score_final, self.active_clients_indicies), reverse=True)])
        print("---check sorted_idx_score : ", sorted_idx)
        self.sort_client[res - 1] = sorted_idx

<<<<<<< HEAD
        print("---check ref_list : ", ref_list)

=======
        print("---check entropy_list : ", entropy_list)
        sorted_entropy = np.array([x for _, x in sorted(zip(entropy_list, self.active_clients_indicies), reverse=True)])
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
        # server_lr = self.server_config['learning_rate'] * (self.server_config['lr_decay_per_round'] ** (round - 1))
        num_participants = len(client_uploads)
        update_direction_state_dict = None
        # agg weights for prototype
        cumsum_per_class = torch.zeros(self.server_config['num_classes']).to(self.clients_dict[0].device)
        agg_weights_vec_dict = {}

        with torch.no_grad():
            for idx, (client_state_dict, prototype_dict, _) in enumerate(client_uploads):
                if self.server_config['FedNH_server_adv_prototype_agg'] == False:
                    cumsum_per_class += prototype_dict['count_by_class_full']
                else:
                    mu = prototype_dict['adv_agg_prototype']
                    W = self.server_model_state_dict['prototype']
                    agg_weights_vec_dict[idx] = torch.exp(torch.sum(W * mu, dim=1, keepdim=True))
                client_update = linear_combination_state_dict(client_state_dict,
                                                              self.server_model_state_dict,
                                                              1.0,
                                                              -1.0,
                                                              exclude=self.exclude_layer_keys
                                                              )
                if idx == 0:
                    update_direction_state_dict = client_update
                else:
                    update_direction_state_dict = linear_combination_state_dict(update_direction_state_dict,
                                                                                client_update,
                                                                                1.0,
                                                                                1.0,
                                                                                exclude=self.exclude_layer_keys
                                                                                )
            # new feature extractor
            self.server_model_state_dict = linear_combination_state_dict(self.server_model_state_dict,
                                                                         update_direction_state_dict,
                                                                         1.0,
                                                                         1 / num_participants,
                                                                         exclude=self.exclude_layer_keys
                                                                         )
            avg_prototype = torch.zeros_like(self.server_model_state_dict['prototype'])
            if self.server_config['FedNH_server_adv_prototype_agg'] == False:
                for _, prototype_dict, _ in client_uploads:
                    avg_prototype += prototype_dict['scaled_prototype'] / cumsum_per_class.view(-1, 1)
            else:
                m = self.server_model_state_dict['prototype'].shape[0]
                sum_of_weights = torch.zeros((m, 1)).to(avg_prototype.device)
                for idx, (_, prototype_dict, _) in enumerate(client_uploads):
                    sum_of_weights += agg_weights_vec_dict[idx]
                    avg_prototype += agg_weights_vec_dict[idx] * prototype_dict['adv_agg_prototype']
                avg_prototype /= sum_of_weights

            # normalize prototype
            avg_prototype = F.normalize(avg_prototype, dim=1)
            # update prototype with moving average
            weight = self.server_config['FedNH_smoothing']
            temp = weight * self.server_model_state_dict['prototype'] + (1 - weight) * avg_prototype
            # print('agg weight:', weight)
            # normalize prototype again
            self.server_model_state_dict['prototype'].copy_(F.normalize(temp, dim=1))
