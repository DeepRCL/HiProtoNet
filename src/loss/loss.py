import torch.nn as nn
import torch
from torchvision.ops import sigmoid_focal_loss
from torchvision.transforms.functional import affine, InterpolationMode
import random
import logging
from src.utils import lorentz as L


class MSE(object):
    def __init__(self, loss_weight=1, reduction="mean"):
        self.criterion = nn.MSELoss(reduction=reduction)
        self.loss_weight = loss_weight
        logging.info(f"setup MSE Loss with loss_weight:{loss_weight}, and reduction:{reduction}")

    def compute(self, pred, target):
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        loss = self.criterion(input=pred, target=target)  # (Num_nodes, 4)
        return self.loss_weight * loss


class CeLoss(object):
    def __init__(self, loss_weight=1, reduction="mean"):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)  # , weight=tbd)
        self.loss_weight = loss_weight
        logging.info(f"setup CE Loss with loss_weight:{loss_weight}, and reduction:{reduction}")

    def compute(self, logits, target):
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        loss = self.criterion(input=logits, target=target)  # (Num_nodes, num_classes)
        return self.loss_weight * loss


class ClusterPatch(object):
    """
    Cluster cost based on ProtoPNet architecture, using distance between patches and prototypes
    """

    def __init__(self, loss_weight, num_classes=4, reduction="mean"):
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.reduction = reduction
        logging.info(
            f"setup Path-Based Cluster Loss with loss_weight:{loss_weight}, for num_classes:{num_classes}, "
            f"and reduction:{reduction}"
        )

    def compute(self, min_distances, target):
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        target_one_hot = nn.functional.one_hot(target, num_classes=self.num_classes)  # shape (N, classes)
        min_distances = min_distances.reshape((min_distances.shape[0], self.num_classes, -1))
        class_specific_min_distances, _ = min_distances.min(dim=2)  # Shape = (N, classes)
        positives = class_specific_min_distances * target_one_hot  # shape (N, classes)

        if self.reduction == "mean":
            loss = positives.sum(dim=1).mean()
        elif self.reduction == "sum":
            loss = positives.sum()

        return self.loss_weight * loss


class SeparationPatch(object):
    """
    Cluster cost based on ProtoPNet architecture, using distance between patches and prototypes
    """

    def __init__(self, loss_weight, num_classes=4, strategy="all", normalize=True, reduction="mean"):
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.strategy = strategy  # calculate loss based on all "closest" negative or "all" negatives for each input
        self.normalize = normalize
        self.reduction = reduction
        logging.info(
            f"setup Patch-Based Separation Loss with loss_weight:{loss_weight}, for num_classes:{num_classes}, \n"
            f"using negatives strategy:{strategy}, with normalization:{normalize} and reduction:{reduction}"
        )

    def compute(self, min_distances, target):
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        target_one_hot = nn.functional.one_hot(target, num_classes=self.num_classes)
        min_distances = min_distances.reshape((min_distances.shape[0], self.num_classes, -1))
        class_specific_min_distances, _ = min_distances.min(dim=2)  # Shape = (N, classes)
        negatives = class_specific_min_distances * (1 - target_one_hot)  # shape (N, classes)

        if self.strategy == "closest":
            # this is based on protopnet paper and results the same, but their code implementation is different
            # We penalize against the closest class of "other" clusters, similar to ProtoPNet's loss equation in paper
            negatives[negatives == 0] = torch.tensor(float('inf')).to(negatives.device)
            negatives, _ = negatives.min(dim=1)  # shape = (N)
        else:
            if self.normalize:
                normalized_negatives = negatives / torch.sum((1 - target_one_hot), dim=1, keepdim=True)
                negatives = normalized_negatives

        if self.reduction == "mean":
            loss = negatives.sum(dim=1).mean()
        elif self.reduction == "sum":
            loss = negatives.sum()

        return (-1) * self.loss_weight * loss  # multiply by -1 to negate loss


class ClusterRoiFeat(object):
    """
    Cluster cost based on XprotoNet architecture, using similarities between ROI features and prototypes
    """

    def __init__(self, loss_weight, num_classes=4, reduction="sum"):
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.reduction = reduction
        logging.info(
            f"setup ROI-Based Cluster Loss with loss_weight:{loss_weight}, for num_classes:{num_classes}, "
            f"and reduction:{reduction}"
        )

    def compute(self, similarities, target):
        """
        compute loss given the similarity scores
        :param similarities: the cosine similarities calculated. shape (N, P). P=num_classes x numPrototypes
        :param target: labels, with shape of (N)
        :return: cluster loss using the similarities between the ROI features and prototypes
        """
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        # turning labels into one hot
        target_one_hot = nn.functional.one_hot(target, num_classes=self.num_classes)  # shape (N, classes)
        # reshaping similarities to group based on class they belong to. shape (N, classes, P_pre_classes)
        similarities = similarities.reshape((similarities.shape[0], self.num_classes, -1))
        # get largest prototype-ROIfeature similarity scores per class
        class_specific_max_similarity, _ = similarities.max(dim=2)  # Shape = (N, classes)
        # pick similarity scores of classes the input belongs to
        positives = class_specific_max_similarity * target_one_hot  # shape (N, classes)
        loss = -1 * positives  # loss is negative sign of similarity scores

        # aggregate loss values
        if self.reduction == "mean":  # average across batch size
            loss = loss.mean(dim=0).sum()
        elif self.reduction == "sum":
            loss = loss.sum()

        return self.loss_weight * loss


class SeparationRoiFeat(object):
    """
    Separation cost based on XprotoNet architecture, using similarities between ROI features and prototypes
    """

    def __init__(self, loss_weight, num_classes=4, strategy="all", normalize=True, reduction="sum", abstain_class=False):
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.strategy = strategy  # calculate loss based on all "closest" negative or "all" negatives for each input
        self.normalize = normalize
        self.reduction = reduction
        self.abstain_class = abstain_class
        logging.info(
            f"setup ROI-Based Separation Loss with loss_weight:{loss_weight}, for num_classes:{num_classes}, "
            f"using negatives strategy:{strategy}, with normalization:{normalize} and reduction:{reduction}"
        )

    def compute(self, similarities, target):
        """
        compute loss given the similarity scores
        :param similarities: the cosine similarities calculated. shape (N, P). P=num_classes x numPrototypes
        :param target: labels, with shape of (N)
        :return: separation loss using the similarities between the ROI features and prototypes
        """
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        # turning labels into one hot
        target_one_hot = nn.functional.one_hot(target, num_classes=self.num_classes)  # shape (N, classes)
        if self.abstain_class:
            # do not calculate the separation losses for the fifth class prototypes
            target_one_hot[:, -1] = 1
        # reshaping similarities to group based on class they belong to. shape (N, classes, P_pre_classes)
        similarities = similarities.reshape((similarities.shape[0], self.num_classes, -1))
        # get largest prototype-ROIfeature similarity scores per class
        class_specific_max_similarity, _ = similarities.max(dim=2)  # Shape = (N, classes)
        # pick similarity scores of classes the input belongs to
        negatives = class_specific_max_similarity * (1 - target_one_hot)  # shape (N, classes)

        if self.strategy == "closest":
            # this is based on protopnet paper and results the same, but their code implementation is different
            # We penalize against the closest class of "other" clusters, similar to ProtoPNet's loss equation in paper
            negatives = negatives.max(dim=1)  # shape = (N)
        else:
            if self.normalize:
                normalized_negatives = negatives / torch.sum((1 - target_one_hot), dim=1, keepdim=True)
                negatives = normalized_negatives

        loss = negatives

        # aggregate loss values
        if self.reduction == "mean":  # average across batch size
            loss = loss.mean(dim=0).sum()
        elif self.reduction == "sum":
            loss = loss.sum()

        return self.loss_weight * loss


class OrthogonalityLoss(object):
    """
    orthogonality loss to encourage diversity in learned prototype vectors
    """

    def __init__(self, loss_weight, num_classes=4, mode="per_class"):
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.mode = mode  # one of 'per_class' or 'all'
        if mode == "per_class":
            self.cosine_similarity = nn.CosineSimilarity(dim=3)
        elif mode == "all":
            self.cosine_similarity = nn.CosineSimilarity(dim=2)
        logging.info(
            f"setup Orthogonality Loss with loss_weight:{loss_weight}, "
            f"for num_classes:{num_classes}, in {mode} mode"
        )

    def compute(self, prototype_vectors):
        """
        compute loss given the prototype_vectors
        :param prototype_vectors: shape (P, channel, 1, 1). P=num_classes x numPrototypes
        :return: orthogonality loss either across each class, summed (or averaged), or across all classes
        """
        if self.loss_weight == 0:
            return torch.tensor(0, device=prototype_vectors.device)

        # per class diversity
        if self.mode == "per_class":
            # reshape to (num_classes, num_prot_per_class, channel):
            prototype_vectors = prototype_vectors.reshape(self.num_classes, -1, prototype_vectors.shape[1])
            # shape of similarity matrix is (num_classes, num_prot_per_class, num_prot_per_class)
            sim_matrix = self.cosine_similarity(prototype_vectors.unsqueeze(1), prototype_vectors.unsqueeze(2))
        elif self.mode == "all":
            # shape of similarity matrix is (num_prot_per_class, num_prot_per_class)
            sim_matrix = self.cosine_similarity(
                prototype_vectors.squeeze().unsqueeze(1),
                prototype_vectors.squeeze().unsqueeze(0),
            )
        # use upper traingle elements of similarity matrix (excluding main diagonal)
        loss = torch.triu(sim_matrix, diagonal=1).sum()

        return self.loss_weight * loss


class HyperbolicOrthogonalityLoss(object):
    """
    orthogonality loss to encourage diversity in learned prototype vectors
    """

    def __init__(self, loss_weight, sim_func, num_classes=4, mode="per_class"):
        self.num_classes = num_classes
        self.sim_func = sim_func
        self.loss_weight = loss_weight
        self.mode = mode  # one of 'per_class' or 'all'
        logging.info(
            f"setup Hyperbolic Orthogonality Loss (with lorentz distance) with loss_weight:{loss_weight}, "
            f"for num_classes:{num_classes}, in {mode} mode"
        )

    def compute(self, prot_x, prot_y, curv, prot_group="part"):
        """
        part_prototypes has shape (P, D) = (c*p, D)
        local_prototypes has shape (PL, D)
        :param prot_x: prototype vector (has to be part-prototype if "mode" is "part_local")
        :param prot_y: prototype vector (has to be local-prototype if "mode" is "part_local")
        :param curv: the hyperboloid curv. may be learnable by the model.
        :param prot_group: groups of prototypes for orthogonality. can be "part", "local", or "part_local"
        :return: orthogonality loss of the selected mode!
        """
        if self.loss_weight[prot_group] == 0:
            return torch.tensor(0, device=curv.device)

        # orthogonality across all prototypes of all classes
        if self.mode == "all":
            prot_distances = L.pairwise_dist(prot_x, prot_y, curv)  # shape (P, P)
            prot_similarity = self.sim_func(prot_distances)
            if prot_group == "part" or prot_group == "local":
                # get the upper triangle for calculation of orthogonality
                mask = torch.triu(torch.ones_like(prot_similarity), diagonal=1)
                loss = (prot_similarity * mask).sum() / mask.sum()
            elif prot_group == "part_local":
                # use all the similarities
                loss = prot_similarity.mean()
        elif self.mode == "per_class":
        # orthogonality across prototypes of the same class
            num_prot_per_class = prot_x.shape[0]//self.num_classes
            if prot_group == "local":
                raise ("cannot perform per_class orthogonality for local prototypes because of having 1_per_class prototypes")
                # loss = torch.tensor(0, device=curv.device)
            elif prot_group == "part_local":
                part_prototypes = prot_x
                local_prototypes = prot_y

                num_g_per_class = local_prototypes.shape[0] // self.num_classes  # L
                local_prototypes = local_prototypes.reshape(num_g_per_class, self.num_classes,
                                                              local_prototypes.shape[1])  # shape (L, c, D)

                num_p_per_class = part_prototypes.shape[0] // self.num_classes  # p
                part_prototypes = part_prototypes.reshape(num_p_per_class, self.num_classes,
                                                          part_prototypes.shape[1])  # shape (p, c, D)

                # we need to entail each part prototype with at least one local prototype of its own class
                # so we repeat_interleave the dim 0 of the part_prototypes and repeat the dim 0 of the local prototype
                # so the result is  (L0,,...,L9,L0,,...,L9,L0,...,L9...)
                # a              nd (p0,p0,...,p0,p1,p1,...,p1,p2,...,p2...)
                # so the distances are (d0,d0,...,d0,d1,d1,...,d1,d2,...,d2...)
                # and we reshape to (p, L, c)
                #                   [(d0,d0,...,d0),
                #                    (d1,d1,...,d1),
                #                    (d2,d2,...,d2)]
                # and then we get the minimum of each row (across dim 1) to get the minimum distances of each broad prototype to its class's local prototypes
                # and then we sum them up to get the total orthogonality loss of each class, and then sum or mean them across classes

                local_prototypes = local_prototypes.repeat(num_p_per_class, 1, 1)  # shape (p*L, c, D)
                part_prototypes = part_prototypes.repeat_interleave(num_g_per_class, dim=0)  # shape (p*L, c, D)

                # elementwise lorentz distance to save compute.
                prot_distances = L.elementwise_dist(part_prototypes, local_prototypes, curv)  # shape (p*L, c)
                prot_similarity = self.sim_func(prot_distances)  # shape (p*L, c)

                # reshape to (p, L, c)
                loss = prot_similarity.reshape(num_p_per_class, num_g_per_class, self.num_classes)
                # get the minimum of each row (dim 1) = orthogonality loss of each broad prototype
                loss = loss.min(dim=1).values  # shape (p, c)
                # average them to get the total orthogonality loss of each class and each batch
                loss = loss.mean()

            elif prot_group == "part":
                prot_x = prot_x.reshape(num_prot_per_class, self.num_classes, prot_x.shape[1])  # shape (p, c, D)
                # need to repeat_interleave the dim 0 to get, (p0,p0,...,p0,p1,p1,...,p1,p2,...,p2...)
                prot_x = prot_x.repeat_interleave(num_prot_per_class, dim=0)  # shape (p*p, c, D)

                prot_y = prot_y.reshape(num_prot_per_class, self.num_classes, prot_y.shape[1])  # shape (p, c, D)
                # need to normal repeat the dim 0 to get, (p0,p1,...,p9,p0,p1,...,p9,p0,...,p9...)
                prot_y = prot_y.repeat(num_prot_per_class, 1, 1)
                prot_distances = L.elementwise_dist(prot_x, prot_y, curv)  # shape (p*p, c)
                prot_similarity = self.sim_func(prot_distances)
                prot_similarity = prot_similarity.reshape(num_prot_per_class, num_prot_per_class, self.num_classes)  # shape (p, p, c)

                # get the upper triangle for calculation of orthogonality
                mask = torch.triu(torch.ones_like(prot_similarity[:,:,0]), diagonal=1).unsqueeze(2)
                loss = (prot_similarity * mask).sum() / (mask.sum() * prot_similarity.shape[-1])

        return self.loss_weight[prot_group] * loss


class L_norm(object):
    def __init__(self, mask=None, p=1, loss_weight=1e-4, reduction="sum"):
        self.mask = mask  # mask determines which elements of tensor to be used for Lnorm calculations
        self.p = p
        self.loss_weight = loss_weight
        self.reduction = reduction
        logging.info(f"setup L{p}-Norm Loss with loss_weight:{loss_weight}, with reduction:{reduction}")

    def compute(self, tensor, dim=None):
        if self.loss_weight == 0:
            return torch.tensor(0, device=tensor.device)

        if self.mask != None:
            loss = (self.mask.to(tensor.device) * tensor).norm(p=self.p, dim=dim)
        else:
            loss = tensor.norm(p=self.p, dim=dim)
        if self.reduction == "mean":
            loss = loss.mean(dim=0).sum()
        elif self.reduction == "sum":
            loss = loss.sum()
        return self.loss_weight * loss


def get_affine_config():
    config = {
        "angle": random.uniform(-20, 20),
        "translate": (
            0,
            0,
        ),
        "scale": random.uniform(0.6, 1.5),
        "shear": 0.0,
        "fill": 0,
        "interpolation": InterpolationMode.BILINEAR,
    }
    return config


class TransformLoss(object):
    """
    the loss applied on generated ROIs!
    """

    def __init__(self, loss_weight=1e-4, reduction="sum"):
        self.loss_weight = loss_weight
        self.criterion = nn.L1Loss(reduction="sum")
        self.reduction = reduction
        logging.info(f"setup Transformation Loss with loss_weight:{loss_weight}, with reduction:{reduction}")

    def compute(self, x, occurrence_map, model, attn_map="part"):
        if self.loss_weight == 0:
            return torch.tensor(0, device=x.device)

        if len(x.shape) == 5:
            video_based = True
        else:
            video_based = False

        # get the affine transform randomly sampled configuration
        config = get_affine_config()

        # transform input and get its new occurrence map
        if video_based:
            N, D, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(-1, D, H, W)  # shape (NxT, D, H, W)
        transformed_x = affine(x, **config)  # shape (NxT, D, H, W), T is 1 for image-based
        if video_based:
            transformed_x = transformed_x.reshape(N, T, D, H, W).permute(0, 2, 1, 3, 4)  # shape (N, D, T, H, W)

        # if feature_group is "part", use attn maps corresponding to the broad features only
        # if feature_group is "both", then use both attn maps, if local_attn_map is not None
        if attn_map == "part":
            occurrence_map_transformed = model.compute_occurence_map(transformed_x).squeeze(2)  # shape (N, P, (T), H, W)
        elif attn_map == "both":
            occurrence_map_transformed, local_attn_map = model.compute_occurence_map(transformed_x)  # shape (N, P, 1, (T), H, W)
            if local_attn_map is not None:
                occurrence_map_transformed = torch.cat([occurrence_map_transformed, local_attn_map], dim=1).squeeze(2)

        # transform initial occurence map
        occurrence_map = occurrence_map.squeeze(2)  # shape (N, P, H, W) or (N, P, T, H, W) for video-based
        if video_based:
            N, P, T, H, W = occurrence_map.shape
            occurrence_map = occurrence_map.permute(0, 2, 1, 3, 4).reshape(-1, P, H, W)  # shape (NxT, P, H, W)
        transformed_occurrence_map = affine(occurrence_map, **config)  # shape (NxT, P, H, W), T is 1 for image-based
        if video_based:
            transformed_occurrence_map = transformed_occurrence_map.reshape(N, T, P, H, W).permute(
                0, 2, 1, 3, 4
            )  # shape (N, P, T, H, W)

        # compute L1 loss
        loss = self.criterion(occurrence_map_transformed, transformed_occurrence_map)
        if self.reduction == "mean":
            loss = loss / (occurrence_map_transformed.shape[0] * occurrence_map_transformed.shape[1])

        return self.loss_weight * loss


class CeLossAbstain(object):
    """
    Cross-entropy-like loss. Introduces a K+1-th class, abstention, which is a
    learned estimate of aleatoric uncertainty. When the network abstains,
    the loss will be computed with the ground truth, but, the network incurs
    loss for using the abstension
    """

    def __init__(self, loss_weight=1, ab_weight=0.3, reduction="sum", ab_logitpath="joined"):
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.ab_weight = ab_weight
        self.ab_logitpath = ab_logitpath
        self.criterion = nn.NLLLoss(reduction=reduction)
        assert self.ab_logitpath == "joined" or self.ab_logitpath == "separate"
        logging.info(
            f"setup CE Abstain Loss with loss_weight:{loss_weight}, "
            + f"ab_penalty:{ab_weight}, ab_path:{ab_logitpath} and reduction:{reduction}"
        )

    def to(self, device):
        self.criterion = self.criterion.to(device)

    def compute(self, logits, target):
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        B, K_including_abs = logits.shape  # (B, K+1)
        K = K_including_abs - 1
        assert K >= 2, "CeLossAbstain input must have >= 2 classes not including abstention"

        # virtual_pred = (1-alpha) * pred + alpha * target
        if self.ab_logitpath == "joined":
            abs_pred = logits.softmax(dim=1)[:, K : K + 1]  # (B, 1)
        elif self.ab_logitpath == "separate":
            abs_pred = logits.sigmoid()[:, K : K + 1]  # (B, 1)
        class_pred = logits[:, :K].softmax(dim=1)  # (B, K)
        target_oh = nn.functional.one_hot(target, num_classes=K)
        virtual_pred = (1 - abs_pred) * class_pred + abs_pred * target_oh  # (B, K)

        loss_pred = self.criterion(torch.log(virtual_pred), target)
        loss_abs = -torch.log(1 - abs_pred).squeeze()  # (B)

        if self.reduction == "mean":
            loss_abs = loss_abs.mean()
        elif self.reduction == "sum":
            loss_abs = loss_abs.sum()

        return self.loss_weight * (loss_pred + self.ab_weight * loss_abs)
    

class Entailment(object):  # TODO Check
    def __init__(self, loss_weight, num_classes=4, reduction="mean"):
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.reduction = reduction
        print(f"setup Entailment Loss with loss_weight:{loss_weight}, with reduction:{reduction}")

    def compute(self, model):
        if self.loss_weight == 0:
            return torch.tensor(0, device=next(model.parameters()).device)

        model.curv.data = torch.clamp(model.curv.data, **model._curv_minmax)
        _curv = model.curv.exp()

        if model.lift_prototypes:
            local_prototypes = L.get_hyperbolic_feats(model.local_prototype_vectors.squeeze(),
                                                       model.visual_alpha, model.curv,
                                                       model.device)  # shape (num_classes, D)
            part_prototypes = L.get_hyperbolic_feats(model.prototype_vectors.squeeze(),
                                                     model.visual_alpha, model.curv, model.device)  # shape (P, D)
        else:
            local_prototypes = model.local_prototype_vectors.squeeze()  # shape (num_classes, D)
            part_prototypes = model.prototype_vectors.squeeze()  # shape (P, D)

        # Option1: for now, we find the entailment of each local_prot with its corresponding part-prots (only positive samples)
        # TODO use pairwise_inner for calculating the inner product of all elements later!

        num_g_per_class = local_prototypes.shape[0] // self.num_classes  # L
        local_prototypes = local_prototypes.reshape(num_g_per_class, self.num_classes,
                                                      local_prototypes.shape[1])  # shape (L, c, D)

        num_p_per_class = part_prototypes.shape[0] // self.num_classes  # p
        part_prototypes = part_prototypes.reshape(num_p_per_class, self.num_classes,
                                                  part_prototypes.shape[1])  # shape (p, c, D)

        # we need to entail each part prototype with at least one local prototype of its own class
        # so we repeat_interleave the dim 0 of the part_prototypes and repeat the dim 0 of the local prototype
        # so the result is  (L0,,...,L9,L0,,...,L9,L0,...,L9...)
        # a              nd (p0,p0,...,p0,p1,p1,...,p1,p2,...,p2...)
        # so the angles are (a0,a0,...,a0,a1,a1,...,a1,a2,...,a2...)
        # and we reshape to (p, L, c)
        #                   [(a0,a0,...,a0),
        #                    (a1,a1,...,a1),
        #                    (a2,a2,...,a2)]
        # and then we get the minimum of each row (dim 0) to get the entailment loss of each broad prototype
        # and then we sum them up to get the total entailment loss of each class, and then sum or mean them across classes

        local_prototypes = local_prototypes.repeat(num_p_per_class, 1, 1)  # shape (p*L, c, D)
        part_prototypes = part_prototypes.repeat_interleave(num_g_per_class, dim=0)  # shape (p*L, c, D)

        _angle = L.oxy_angle(local_prototypes, part_prototypes, _curv)  # shape (P)
        _aperture = L.half_aperture(local_prototypes, _curv)  # shape (P)
        entailment_loss = torch.clamp(_angle - _aperture, min=0)  # shape (P)

        # reshape to (p, L, c)
        entailment_loss = entailment_loss.reshape(num_p_per_class, num_g_per_class, self.num_classes)
        # get the minimum of each row (dim 1) = entailment loss of each broad prototype
        entailment_loss = entailment_loss.min(dim=1).values  # shape (p, c)
        # sum them up to get the total entailment loss of each class
        entailment_loss = entailment_loss.sum(dim=0)
        # sum or mean them across classes
        if self.reduction == "mean":
            entailment_loss = entailment_loss.mean()
        elif self.reduction == "sum":
            entailment_loss = entailment_loss.sum()
        return self.loss_weight * entailment_loss

