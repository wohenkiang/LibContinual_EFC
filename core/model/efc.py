import torch
import itertools
from torch import nn
import numpy as np
from copy import deepcopy
from torch.distributions import MultivariateNormal
import os

# import einops  # 注释掉，不再需要

def isPSD(A, tol=1e-7):
    import numpy as np
    A = A.cpu().numpy()
    E = np.linalg.eigvalsh(A)
    print("Maximum eigenvalue {}".format(np.max(E)))
    return np.all(E > -tol)


class EmpiricalFeatureMatrix:

    def __init__(self, device):
        self.empirical_feat_mat = None
        self.device = device

    def create(self, model):
        self.empirical_feat_mat = torch.zeros((model.get_feat_size(), model.get_feat_size()), requires_grad=False).to(
            self.device)

    def get(self):
        return self.empirical_feat_mat

    def compute(self, model, trn_loader, task_id):
        self.compute_efm(model, trn_loader, task_id)

    def compute_efm(self, model, trn_loader, task_id):
        print("Evaluating Empirical Feature Matrix")
        # Compute empirical feature matrix for specified number of samples -- rounded to the batch size

        n_samples_batches = len(trn_loader.dataset) // trn_loader.batch_size
        print("EFM len(trn_loader.dataset) ;trn_loader.batch_size; n_samples_batches")
        print(len(trn_loader.dataset),trn_loader.batch_size,n_samples_batches)
        model.eval()
        # ensure that gradients are zero
        model.zero_grad()

        self.create(model)

        with torch.no_grad():
            for batch in itertools.islice(trn_loader, n_samples_batches):
                print(batch)
                images = batch["image"]
                print(images.shape)
                label = batch["label"]
                gap_out = model.backbone(images.to(self.device))['features']

                out = torch.cat([h(gap_out) for h in model.heads], dim=1)

                out_size = out.shape[1]

                # compute the efm using the closed formula
                log_p = nn.LogSoftmax(dim=1)(out)

                identity = torch.eye(out.shape[1], device=self.device)

                # 原始 einops 实现（注释掉）
                # der_log_softmax =  einops.repeat(identity, 'n m -> b n m', b=gap_out.shape[0]) - einops.repeat(torch.exp(log_p), 'n p -> n a p', a=out.shape[1])

                # TODO
                part1 = identity.unsqueeze(0).expand(gap_out.shape[0], identity.shape[0], identity.shape[1])
                part2 = torch.exp(log_p).unsqueeze(1).expand(log_p.shape[0], out.shape[1], log_p.shape[1])
                der_log_softmax = part1 - part2

                weight_matrix = torch.cat([h[0].weight for h in model.heads], dim=0)

                weight_matrix_expanded = weight_matrix.unsqueeze(0).expand(gap_out.shape[0], weight_matrix.shape[0],
                                                                           weight_matrix.shape[1])
                jac = torch.bmm(der_log_softmax, weight_matrix_expanded)

                efm_per_batch = torch.zeros((images.shape[0], model.get_feat_size(), model.get_feat_size()),
                                            device=self.device)

                p = torch.exp(log_p)

                # jac =  jacobian_in_batch(log_p, gap_out).detach() # equivalent formulation with gradient computation, with torch.no_grad() should be removed

                for c in range(out_size):
                    efm_per_batch += p[:, c].view(images.shape[0], 1, 1) * torch.bmm(
                        jac[:, c, :].unsqueeze(1).permute(0, 2, 1), jac[:, c, :].unsqueeze(1))

                self.empirical_feat_mat += torch.sum(efm_per_batch, dim=0)

        n_samples = n_samples_batches * trn_loader.batch_size

        # divide by the total number of samples
        self.empirical_feat_mat = self.empirical_feat_mat / n_samples

        if isPSD(self.empirical_feat_mat):
            print("EFM is semidefinite positive")

        return self.empirical_feat_mat

class ProtoGenerator:

    def __init__(self, device, task_dict, batch_size, feature_space_size) -> None:

        self.device = device
        self.task_dict = task_dict
        self.batch_size = batch_size
        self.feature_space_size = feature_space_size
        self.prototype = []
        self.variances = []
        self.class_label = []

        self.R = None
        self.running_proto = None
        self.running_proto_variance = []
        self.rank = None
        self.current_mean = None
        self.current_std = None
        self.gaussians = {}
        self.rank_list = []
        self.class_stats = {}

    def compute(self, model, loader, current_task):
        model.eval()

        features_list = []
        label_list = []

        with torch.no_grad():
            for batch in loader:
                images = batch["image"]
                labels = batch["label"]
                images = images.to(self.device)
                labels = labels.type(dtype=torch.int64).to(self.device)
                _, features = model(images)

                label_list.append(labels.cpu())
                features_list.append(features.cpu())

        label_list = torch.cat(label_list)
        features_list = torch.cat(features_list)
        print(label_list)
        for label in self.task_dict[current_task]:
            mask = (label_list == label)
            feature_classwise = features_list[mask]
            self.class_stats[label] = feature_classwise.shape[0]

            proto = feature_classwise.mean(dim=0)

            covariance = torch.cov(feature_classwise.T)

            self.running_proto_variance.append(covariance)
            self.prototype.append(proto)
            self.class_label.append(label)
            self.gaussians[label] = MultivariateNormal(
                proto.cpu(),
                covariance_matrix=covariance + 1e-5 * torch.eye(covariance.shape[0]).cpu(),
            )

        self.running_proto = deepcopy(self.prototype)

    def update_gaussian(self, proto_label, mean, var):
        self.gaussians[proto_label] = MultivariateNormal(
            mean.cpu(),
            covariance_matrix=var + 1e-5 * torch.eye(var.shape[0]).cpu(),
        )

    def perturbe(self, current_task, protobatchsize=64):

        # list of number of classes seen before

        index = list(range(0, sum([len(self.task_dict[i]) for i in range(0, current_task)])))
        np.random.shuffle(index)

        proto_aug_label = torch.LongTensor(self.class_label)[index]

        if len(self.running_proto) < protobatchsize:
            samples_to_add = protobatchsize - len(self.running_proto)
            proto_aug_label = torch.cat([proto_aug_label,
                                         proto_aug_label.repeat(int(np.ceil(samples_to_add / len(self.running_proto))))[
                                         :samples_to_add]])
        else:
            proto_aug_label = proto_aug_label[:protobatchsize]

        proto_aug_label, _ = torch.sort(proto_aug_label)
        samples_to_generate = torch.nn.functional.one_hot(proto_aug_label).sum(dim=0)
        proto_aug = []
        for class_idx, n_samples in enumerate(samples_to_generate):
            if n_samples > 0:
                proto_aug.append(self.gaussians[class_idx].sample((n_samples,)))

        proto_aug = torch.cat(proto_aug, dim=0)
        n_proto = proto_aug.shape[0]
        shuffle_indexes = torch.randperm(n_proto)
        proto_aug = proto_aug[shuffle_indexes, :]
        proto_aug_label = proto_aug_label[shuffle_indexes]

        return proto_aug, proto_aug_label, n_proto
    def update_task_dict(self,task_dict):
        self.task_dict = task_dict

def compute_rotations(images, image_size, task_dict, targets, task_id):
    # compute self-rotation for the first task following PASS https://github.com/Impression2805/CVPR21_PASS
    images_rot = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(1, 4)], 1)
    images_rot = images_rot.view(-1, 3, image_size, image_size)
    if task_dict is not None:
        target_rot = torch.stack([(targets * 3 + k) + len(task_dict[task_id]) - 1 for k in range(1, 4)], 1).view(-1)
    else:
        target_rot=None
    return images_rot, target_rot



def get_old_new_features(model, old_model, trn_loader, device):
    model.eval()
    old_model.eval()

    features_list = []
    old_features_list = []
    labels_list = []
    old_outputs = []
    with torch.no_grad():
        for  batch in trn_loader:
            images, labels = batch['image'], batch['label']
            images = images.to(device)
            labels = labels.type(dtype=torch.int64).to(device)
            _, features = model(images)
            old_out, old_features = old_model(images)
            old_outputs.append(torch.cat(list(old_out.values()), dim=1))
            features_list.append(features)
            old_features_list.append(old_features)
            labels_list.append(labels)

        old_outputs = torch.cat(old_outputs, dim=0)
        old_features = torch.cat(old_features_list)
        new_features = torch.cat(features_list)
        labels_list = torch.cat(labels_list)

    return new_features, old_features

from .finetune import Finetune

class BaseModel(nn.Module):
    def __init__(self, backbone, feat_dim):
        super(BaseModel, self).__init__()
        self.backbone = backbone
        self.heads = nn.ModuleList()
        self.feat_dim = feat_dim

    def add_classification_head(self, n_out):

        self.heads.append(
            torch.nn.Sequential(nn.Linear(512, n_out, bias=False)))

    def get_feat_size(self):
        return self.feat_dim

    def forward(self, x):
        results = {}
        features = self.backbone(x)['features']
        for id, head in enumerate(self.heads):
            results[id] = head(features)

        return results, features

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all parameters from the main model, but not the heads"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

class EFC(Finetune):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        self.kwargs = kwargs
        self.model = BaseModel(backbone,feat_dim)
        self.old_model = None
        self.efc_lamb = kwargs['efc_lamb']
        self.damping = kwargs['efc_damping']
        self.sigma_proto_update = kwargs['efc_protoupdate']
        self.protobatch_size = kwargs['efc_protobatchsize']
        self.task_dict = {}

        self.proto_generator = ProtoGenerator(device=kwargs['device'],
                                              batch_size=kwargs['batch_size'],task_dict=self.task_dict,
                                              feature_space_size=self.model.feat_dim,
                                              )

        self.matrix_rank = None
        self.R = None
        self.auxiliary_classifier = None
        self.previous_efm = None
        self.batch_idx = 0
        self.task_id = 0

        self.device = kwargs['device']
        self.inc_cls_num = kwargs['inc_cls_num']
        self.init_cls_num = kwargs['init_cls_num']

        self.batch_size = kwargs['batch_size']
        self.milestones_first_task = None
        self.dataset = kwargs['dataset']
        if self.dataset == "cifar100":
            self.image_size = 32
        elif self.dataset == "tiny-imagenet":
            self.image_size = 64

        # SELF-ROTATION classifier
        self.auxiliary_classifier = nn.Linear(512, self.init_cls_num* 3)

    def efm_loss(self, features, features_old):
        features = features.unsqueeze(1)
        features_old = features_old.unsqueeze(1)
        matrix_reg = self.efc_lamb * self.previous_efm + self.damping * torch.eye(self.previous_efm.shape[0],
                                                                                  device=self.device)
        efc_loss = torch.mean(
            torch.bmm(torch.bmm((features - features_old), matrix_reg.expand(features.shape[0], -1, -1)),
                      (features - features_old).permute(0, 2, 1)))
        return efc_loss

    def rescale_targets(self, targets, t):
        offset = (t - 1) * self.inc_cls_num + self.init_cls_num if self.init_cls_num > -1 and t > 0 else t * self.inc_cls_num
        targets = targets - offset
        return targets

    def train_criterion(self, outputs, targets, t, features, old_features, proto_to_samples, current_batch_size):

        cls_loss, loss_protoAug, reg_loss, n_proto = 0, 0, 0, 0

        if t > 0:
            # EFM loss
            reg_loss = self.efm_loss(features[:current_batch_size], old_features)

            with torch.no_grad():
                # prototype sampling
                proto_aug, proto_aug_label, n_proto = self.proto_generator.perturbe(t, self.protobatch_size)
                proto_aug = proto_aug[:proto_to_samples].to(self.device)
                proto_aug_label = proto_aug_label[:proto_to_samples].to(self.device)
                n_proto = proto_to_samples

            soft_feat_aug = []
            for _, head in enumerate(self.model.heads):
                soft_feat_aug.append(head(proto_aug))

            soft_feat_aug = torch.cat(soft_feat_aug, dim=1)
            overall_logits = torch.cat(
                [soft_feat_aug, torch.cat(list(outputs.values()), dim=1)[current_batch_size:, :]], dim=0)
            overall_targets = torch.cat([proto_aug_label, targets[current_batch_size:]])

            # loss over all encountered classes (proto+current class samples)
            loss_protoAug = torch.nn.functional.cross_entropy(overall_logits, overall_targets)

            # loss over current classes (only current class samples)
            cls_loss = torch.nn.functional.cross_entropy(outputs[t][:current_batch_size, :],
                                                         self.rescale_targets(targets[:current_batch_size], t))

        else:
            # first task only a cross entropy loss
            cls_loss = torch.nn.functional.cross_entropy(torch.cat(list(outputs.values()), dim=1), targets)

        return cls_loss, reg_loss, loss_protoAug, n_proto

    def tag_probabilities(self, outputs):
        tag_output = []
        for key in outputs.keys():
            tag_output.append(outputs[key])
        tag_output = torch.cat(tag_output, dim=1)
        probabilities = torch.nn.Softmax(dim=1)(tag_output)
        return probabilities

    def taw_probabilities(self, outputs, head_id):
        probabilities = torch.nn.Softmax(dim=1)(outputs[head_id])
        return probabilities

    def update_task_idt(self, dataloader):
        all_labels = []

        for b, batch in enumerate(dataloader):
            labels = batch['label']
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()

            all_labels.extend(labels)

        unique_labels = list(set(all_labels))
        self.task_dict[self.task_id] = unique_labels
        print(self.task_dict,type(unique_labels[0]))
        
    def observe(self, data):
        if self.task_id == 0:
            self.model.train()
        else:
            # we freeze batch norm after the first task.
            self.model.eval()
        if  self.task_id == 0 :
            self.auxiliary_classifier.train()
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)

        current_batch_size = x.shape[0]

        if self.task_id == 0:
            images_rot, target_rot = compute_rotations(x, self.image_size, self.task_dict, y, self.task_id)
            x = torch.cat([x, images_rot], dim=0)
            y = torch.cat([y, target_rot], dim=0)

        if self.task_id > 0:
            # Forward old model
            _, old_features = self.old_model(x)

            if self.buffer.previous_batch_samples is not None:
                # Sample from a buffer of current task data for PR-ACE
                pb, previous_batch_samples, previous_batch_labels = self.buffer.sample()
                x = torch.cat([x, previous_batch_samples], dim=0)
                y = torch.cat([y, previous_batch_labels], dim=0)
            else:
                pb = self.protobatch_size
        else:
            old_features = None
            pb = 0

        # Forward in the current model
        outputs, features = self.model(x)

        if self.task_id == 0 :
            # predict the rotation the rotations
            out_rot = self.auxiliary_classifier(features)
            outputs[0] = torch.cat([outputs[0], out_rot], axis=1)

        # compute criterion
        cls_loss, efc_loss, loss_protoAug, _ = self.train_criterion(outputs, y, self.task_id, features, old_features,
                                                                    current_batch_size=current_batch_size,
                                                                    proto_to_samples=pb)

        loss = cls_loss + efc_loss + loss_protoAug
        if self.task_id > 0:
            self.buffer.add_data(current_samples=x[:current_batch_size],
                                 current_targets=y[:current_batch_size])

        tag_probs = self.tag_probabilities(outputs)
        pred = torch.argmax(tag_probs, dim=1)
        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0), loss

    def inference(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)

        outputs, features = self.model(x)
        tag_probs = self.tag_probabilities(outputs)
        pred = torch.argmax(tag_probs, dim=1)

        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0)

    def forward(self, x):
        raise ValueError("这是一个值错误")
        if self.task_id == 0:
            images_rot, _ = compute_rotations(x, self.image_size, self.task_dict, None, self.task_id)
            x = torch.cat([x, images_rot], dim=0)

        if self.task_id > 0:
            # Forward old model
            _, old_features = self.old_model(x)

            if self.buffer.previous_batch_samples is not None:
                # Sample from a buffer of current task data for PR-ACE
                pb, previous_batch_samples, previous_batch_labels = self.buffer.sample()
                x = torch.cat([x, previous_batch_samples], dim=0)
            else:
                pb = self.protobatch_size
        else:
            old_features = None
            pb = 0

        # Forward in the current model
        outputs, features = self.model(x)

        if self.task_id == 0:
            # predict the rotation the rotations
            out_rot = self.auxiliary_classifier(features)
            outputs[0] = torch.cat([outputs[0], out_rot], axis=1)
        return outputs

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        self.task_id = task_idx
        self.update_task_idt(train_loader)
        self.proto_generator.update_task_dict(self.task_dict)

        if self.task_id == 0 :
            # Auxiliary classifier used for self rotation
            self.auxiliary_classifier.to(self.device)
            self.auxiliary_classifier.train()

            # Freeze old model
        self.old_model = deepcopy(self.model)
        self.old_model.freeze_all()
        self.old_model.to(self.device)
        self.old_model.eval()

        # Add classification head
        self.model.add_classification_head(len(self.task_dict[self.task_id]))
        self.model.to(self.device)

        if self.task_id > 0:
            # initialize the buffer for PR-ACE for each new tasks
            print("Using PR-ACE")
            self.buffer = buffer
            self.buffer.get_seen_and_current_class_num(self.task_id,self.task_dict)
        else:
            print("Standard training with cross entropy")

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        """Runs after training all the epochs of the task (after the train session)"""

        with torch.no_grad():
            if task_idx > 0 and self.sigma_proto_update != -1:
                new_features, old_features = get_old_new_features(self.model, self.old_model,
                                                                  train_loader, self.device)

                drift = self.compute_drift(new_features, old_features, device="cpu")
                drift = drift.cpu()
                for i, (p, var, proto_label) in enumerate(zip(self.proto_generator.prototype,
                                                              self.proto_generator.running_proto_variance,
                                                              self.proto_generator.class_label)):
                    mean = p + drift[i]
                    self.proto_generator.update_gaussian(proto_label, mean, var)
                    # final update the mean
                    self.proto_generator.prototype[i] = mean

                self.proto_generator.running_proto = deepcopy(self.proto_generator.prototype)

            efm_matrix = EmpiricalFeatureMatrix(self.device)
            efm_matrix.compute(self.model, deepcopy(train_loader), task_idx)
            self.previous_efm = efm_matrix.get()
            R, L, V = torch.linalg.svd(self.previous_efm)
            matrix_rank = torch.linalg.matrix_rank(self.previous_efm)
            print("Computed Matrix Rank {}".format(matrix_rank))
            self.R = R
            self.L = L
            self.matrix_rank = matrix_rank

        # save matrix after each task for analysis

        self.proto_generator.compute(self.model, deepcopy(train_loader), task_idx)

    def compute_drift(self, new_features, old_features, device):
        DY = (new_features - old_features).to(device)
        new_features = new_features.to(device)
        old_features = old_features.to(device)
        running_prototypes = torch.stack(self.proto_generator.running_proto, dim=0)

        running_prototypes = running_prototypes.to(device)
        distance = torch.zeros(len(running_prototypes), new_features.shape[0])

        for i in range(running_prototypes.shape[0]):
            # we use the EFM to update prototypes
            curr_diff = (old_features - running_prototypes[i, :].unsqueeze(0)).unsqueeze(1).to(self.device)

            distance[i] = -torch.bmm(torch.bmm(curr_diff, self.previous_efm.expand(curr_diff.shape[0], -1, -1)),
                                     curr_diff.permute(0, 2, 1)).flatten().cpu()

        scaled_distance = (distance - distance.min()) / (distance.max() - distance.min())

        W = torch.exp(scaled_distance / (2 * self.sigma_proto_update ** 2))
        normalization_factor = torch.sum(W, axis=1)[:, None]
        W_norm = W / torch.tile(normalization_factor, [1, W.shape[1]])

        displacement = torch.zeros((running_prototypes.shape[0], 512))

        for i in range(running_prototypes.shape[0]):
            displacement[i] = torch.sum((W_norm[i].unsqueeze(1) * DY), dim=0)

        return displacement

    def get_parameters(self, config):
        train_parameters = []
        if self.task_id == 0:
            params_to_optimize = [p for p in self.model.backbone.parameters() if p.requires_grad] + [p for p in
                                                                                                self.model.heads.parameters()
                                                                                                if p.requires_grad]
            params_to_optimize += [p for p in self.auxiliary_classifier.parameters() if p.requires_grad]
            train_parameters.append({"params": params_to_optimize})
        else:
            self.model.freeze_bn()
            old_head_params = [p for p in self.model.heads[:-1].parameters() if p.requires_grad]
            new_head_params = [p for p in self.model.heads[-1].parameters() if p.requires_grad]
            head_params = old_head_params + new_head_params
            backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
            params_to_optimize = backbone_params + head_params
            train_parameters.append({"params": params_to_optimize})
        return train_parameters