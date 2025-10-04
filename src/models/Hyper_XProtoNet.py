import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.receptive_field import compute_proto_layer_rf_info_v2
from src.models.ProtoPNet import PPNet, base_architecture_to_features
from src.models.XProtoNet import XProtoNet
from src.utils import lorentz as L
from src.utils.model_utils import get_prototype_class_identity


class Hyper_XProtoNet(XProtoNet):
    def __init__(self,
                 num_local_prototypes_per_class: int = 1,
                 curv_init: float = 1.0,
                 learn_curv: bool = False,
                 init_weights=True,
                 cls_method: str = "broad",
                 local_prototype_method: str = "attn",  # can be "attn" or "last_layer
                 local_feat_method: str = "1_per_input",  # can be "1_per_input" or "1_per_class" TODO try both
                 lift_prototypes: bool = False,
                 local_prot_donut: float = 1.0,
                 # pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),  # TODO modify
                 # pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),  # TODO modify
                 **kwargs):
        super(Hyper_XProtoNet, self).__init__(init_weights=False, **kwargs)

        ###############################################################################
        ############## Extract higher level feature for the hierarchy #################
        # TODO alternatives:
        #  final feature of CNN (can be visualized with grad-cam style stuff)
        #  <CLS> token of transformer. (can show the attention map with rollout)
        # Option 1: using attention map

        ##########################################################
        # number local features to be extracted per case!
        self.local_feat_method = local_feat_method
        if self.local_feat_method == "1_per_input":
            self.num_local_features = 1
        elif self.local_feat_method == "1_per_class":
            self.num_local_features = self.num_classes
        else:
            raise "Invalid local_feat_method passed in!"

        ##########################################################
        # local prototypes!
        # dimension of the local prototypes is the same as the dimension of the part prototypes but the number of local prototypes is different
        self.num_local_prototypes_per_class = num_local_prototypes_per_class
        self.num_local_prototypes = num_local_prototypes_per_class * self.num_classes
        self.local_prototype_shape = (self.num_local_prototypes, self.prototype_shape[1], 1, 1)
        # shape  (PL=num_classes*L, D, 1, 1)
        self.local_prototype_vectors = nn.Parameter(torch.__dict__[self.prot_init_method](self.localprototype_shape)*glocalprot_donut,
                                                     requires_grad=True)
        self.local_prototype_class_identity = get_prototype_class_identity(num_prototypes=self.num_local_prototypes,
                                                                            num_classes=self.num_classes)
        ##########################################################

        # local prototype creation method
        self.local_prototype_method = local_prototype_method
        # make sure last-layer based method is not active at the same time as having 1_per_class local features
        if self.local_prototype_method == "last_layer":
            assert (self.local_feat_method != "1_per_class")

        self.num_local_attn_map = self.num_local_features
        # attention map for the local feature (foreground object)
        cnn_backbone_out_channels = self.get_cnn_backbone_out_channels(self.cnn_backbone)
        self.local_attn_module = nn.Sequential(
            nn.Conv2d(
                in_channels=cnn_backbone_out_channels,
                out_channels=self.prototype_shape[1],
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.prototype_shape[1],
                out_channels=self.prototype_shape[1] // 2,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.prototype_shape[1] // 2,
                out_channels=self.num_local_attn_map,
                kernel_size=1,
                bias=False,
            ),
            # TODO ADD RELU? so only positive attention is created? or abs value is ok?
        )

        # Option 2: Use features directly from the last layer of the CNN encoder
        # Need global pooling to get it to 1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Reduces (7, 7) to (1, 1)
        # commented this out because technically local feature should come from the already extracted features?
        # self.local_conv_1x1 = nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1)

        ###############################################################################
        # Last classification layer, redefine to initialize randomly
        # TODO classify based on both part-level and local-level prototype activations
        self.cls_method = cls_method
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)  # do not use bias
        self.last_layer_local = nn.Linear(self.num_local_prototypes, self.num_classes, bias=False)  # do not use bias

        ###############################################################
        ########## MERU-based hyperbolic parameters ###############
        # # Initialize a learnable logit scale parameter.
        # TODO check if it is useful here. seems to be related to temperature used in cross entropy loss in CLIP/MERU
        # self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

        # # Color mean/std to normalize image.
        # self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1))
        # self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1))

        # TODO include if DDP is used
        # # Get rank of current GPU process for gathering features.
        # self._rank = dist.get_rank()

        # Initialize curvature parameter. Hyperboloid curvature will be `-curv`.
        self.curv = nn.Parameter(
            torch.tensor(curv_init).log(), requires_grad=learn_curv
        )
        # When learning the curvature parameter, restrict it in this interval to
        # prevent training instability.
        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }

        # Learnable scalars to ensure that image features have an expected
        # unit norm before exponential map (at initialization).
        self.visual_alpha = nn.Parameter(torch.tensor(self.prototype_shape[1] ** -0.5).log())
        self.set_alpha = False  # to initialize the alpha value based on the extracted features! this flag is set at the beginning of the training once!
        # TODO maybe similar to MERU that had visual and text alphas,
        #  we need alphas for diff hierarchical layers?
        ##########################################################
        # to lift prototype vectors to hyperboloid (thus assuming they are in euclidean space)   OR
        # not to lift them, thus assuming they are already in hyperbolic space and are learnt to be on the hyperboloid
        self.lift_prototypes = lift_prototypes

        if init_weights:
            self._initialize_weights(self.add_on_layers)
            self._initialize_weights(self.occurrence_module)
            self._initialize_weights(self.local_attn_module)
            if (self.cls_method == "broad") or (self.cls_method == "both"):
                self.set_last_layer_incorrect_connection(layer_name="last_layer", incorrect_strength=0)
            if (self.cls_method == "local") or (self.cls_method == "both"):
                self.set_last_layer_incorrect_connection(layer_name="last_layer_local", incorrect_strength=0)

    def forward(self, x):
        (_, part_feat_prot_lorentz_distance, occurrence_map,
         _, local_feat_prot_lorentz_distance, local_attn_map, logits) = self.forward_detailed(x)
        return logits, part_feat_prot_lorentz_distance, occurrence_map, local_feat_prot_lorentz_distance, local_attn_map

    def distance_2_similarity(self, distances, max_distance=0):
        # TODO CHECK TO FIND WITH WHAT FORMULA THE HYPERBOLIC DISTANCE CAN BE CONVERTED TO SIMILARITY SCORE!
        if self.prototype_activation_function == "log":
            return torch.log((distances + 1) / (distances + self.epsilon))
        else:
            return -distances

    def extract_features(self, x):
        logits, part_feat_prot_lorentz_distance, _, _ = self.forward(x)
        prototype_activations = self.distance_2_similarity(part_feat_prot_lorentz_distance)  # shape (N, P)

        # shape (N, C)             (N, P)                   (N, P)
        return logits, part_feat_prot_lorentz_distance, prototype_activations

    def compute_occurence_map(self, x):
        # Feature Extractor Layer
        x = self.cnn_backbone(x)
        occurrence_map = self.get_occurence_map(x)  # shape (N, P, 1, H, W)
        local_attn_map = self.get_local_attn_map(x)  # shape (N, L, 1, H, W)
        return occurrence_map, local_attn_map

    def get_local_attn_map(self, x):
        # shape (N, L, H, W)   L=num_local_prots
        if self.local_prototype_method == "last_layer":
            # local_attn_map is None because we need to use gradcam etc to visualize what the local feature means!
            local_attn_map = None
        elif self.local_prototype_method == "attn":
            local_attn_map = self.localattn_module(x)  # shape (N, L, H, W)

            # # Option 1: Absolute value, TODO double check again
            # localattn_map = torch.abs(localattn_map).unsqueeze(2)  # shape (N, L, 1, H, W)

            # Option 2: Softmax!
            # n, L, h, w = localattn_map.shape
            # localtn_map = glocalltn_map.reshape((n, L, -1))
            # local_attn_map = self.om_softmax(local_attn_map).reshape((n, L, h, w)).unsqueeze(2)  # shape (N, P, 1, H, W)

            # Option 3: Layernorm + sigmoid!  BAD
            # N, L, H, W = local_attn_map.shape
            # local_attn_map = local_attn_map.view(N * L, H, W)  # Flatten N and P into a single dimension
            # localattn_map = F.layer_norm(localattn_map, (H, W))  # Normalize over H and W
            # localattn_map = localattn_map.view(N, L, H, W)  # Reshape back to original dimensions
            # localattn_map = F.sigmoid(localattn_map)
            # localattn_map = localattn_map.unsqueeze(2)  # shape (N, L, 1, H, W)

            # Option 4: Layernorm!  BAD
            # N, L, H, W = local_attn_map.shape
            # local_attn_map = local_attn_map.view(N * L, H, W)  # Flatten N and P into a single dimension
            # local_attn_map = F.layer_norm(local_attn_map, (H, W))  # Normalize over H and W
            # local_attn_map = local_attn_map.view(N, L, H, W)  # Reshape back to original dimensions
            # local_attn_map = local_attn_map.unsqueeze(2)  # shape (N, L, 1, H, W)

            # Option 5: sigmoid!  TODO double check again
            # local_attn_map = F.sigmoid(local_attn_map).unsqueeze(2)  # shape (N, L, 1, H, W)

            # Option 6: min-max norm! subtract min, divide by max! to be in range [0-1]
            local_attn_map = local_attn_map - local_attn_map.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
            epsilon = 1e-8
            local_attn_map = local_attn_map / (epsilon + local_attn_map.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0])
            local_attn_map = local_attn_map / (epsilon + local_attn_map.sum(dim=(2,3), keepdim=True))
            local_attn_map = local_attn_map.unsqueeze(2)  # shape (N, L, 1, T, H, W)

        else:
            raise "Invalid local prototype method passed in!"

        return local_attn_map

    @property
    def device(self) -> torch.device:
        return self.curv.device

    def push_forward(self, x):
        # (features_to_track, part_feat_prot_lorentz_distance, occurrence_map,
        #  local_features_to_track, local_feat_prot_lorentz_distance, local_attn_map, logits)
        return self.forward_detailed(x)

    def forward_detailed(self, x):
        """
        this method contains all the details of forward propagation and returns needed content!
        """
        ###############################################################
        ######### extract Euclidean features  #########
        ### part-level features
        x = self.cnn_backbone(x)
        feature_map = self.add_on_layers(x).unsqueeze(1)  # shape (N, 1, D, H, W)
        occurrence_map = self.get_occurence_map(x)  # shape (N, P, 1, H, W)

        features_extracted = (occurrence_map * feature_map).sum(dim=3).sum(dim=3)  # shape (N, P, D)
        # TODO need to do average! not sum! eh?
        # features_extracted = (occurrence_map * feature_map).mean(dim=(3, 4))  # shape (N, P, D)

        ### local-level features
        if self.local_prototype_method == "attn":
            local_attn_map = self.get_local_attn_map(x)  # shape (N, L, 1, H, W)
            local_feats_extracted = (local_attn_map * feature_map).sum(dim=3).sum(dim=3)  # shape (N, L, D)
            # TODO need to do average! not sum! eh?
            # local_feats_extracted = (local_attn_map * feature_map).mean(dim=(3, 4))  # shape (N, L, D)
            if self.local_feat_method == "1_per_class":
                # Need to find the distance of each feature to prototypes of its class,
                # Will not reduce it with maxpool, because FC layer relies on all prototype activations
                # in other words, 2 local-prot-per-class = [1,1,2,2,3,3]
                #                 1 local-feat-per-class = [1,  2,  3  ]
                # So, we need repeat_interleave to make it[1,1,2,2,3,3]
                # interleave on dim 1, to expand the shape to (N, L*local_prot_per_class, D)
                local_feats_extracted = local_feats_extracted.repeat_interleave(
                    self.num_local_prototypes_per_class,
                    dim=1)
        elif self.local_prototype_method == "last_layer":  # TODO Fix to make it compatible with the new code
            local_feats_extracted = self.global_avg_pool(feature_map).reshape((-1,1,self.prototype_shape[1]))   # shape (N, L=1, D)
            # commented it out because we technically find local feature by using all the extracted features?
            # local_feats_extracted = self.local_conv_1x1(local_feats_extracted).squeeze()   # shape (N, L, D)
        else:
            raise "Invalid local prototype method passed in!"
        ###############################################################

        ###############################################################
        ##### MERU-based operations for hyperbolic space analysis #####
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()
        # Clamp scaling factors such that they do not up-scale the feature norms.
        # Once `exp(scale) = 1`, they can simply be removed during inference.
        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)

        # initialize the alpha value based on the features_extracted
        if self.set_alpha == True:
            # find the norm of the features_extracted and then use it to initialize the alpha value
            norm = torch.norm(features_extracted, dim=2).detach()
            self.visual_alpha.data = torch.log(1/(norm.mean()))

            # scale up the prototype values to have the same norm as features_extracted
            norm_local = torch.norm(self.local_prototype_vectors, dim=1).detach()
            self.local_prototype_vectors.data = self.local_prototype_vectors.data * (norm.mean() / norm_local.mean())

            norm_broad = torch.norm(self.prototype_vectors, dim=1).detach()
            self.prototype_vectors.data = self.prototype_vectors.data * (norm.mean() / norm_broad.mean())

            self.set_alpha = False

        ###########################################################################################################
        # lift features to hyperbolic space
        hyperbolic_feature_map = L.get_hyperbolic_feats(features_extracted,
                                                        self.visual_alpha, self.curv, self.device)  # shape (N, P, D)
        hyperbolic_local_feats = L.get_hyperbolic_feats(local_feats_extracted,
                                                         self.visual_alpha, self.curv, self.device)  # shape (N, L, D)
        prototype_vectors, local_prototype_vectors = self.get_prototype_vectors()

        ###########################################################################################################
        # Option 1: don't lift prototypes to hyperbolic space. They need to be learned to be on hyperboloid.
        #       later, they get updated with hyperbolic features
        # We implement Option 1 for now
        # TODO Option 2
        # Option 2: lift prototypes to hyperbolic space as well. later, they get updated with euclidean vectors!
        # TODO CHECK IF THIS IS NECESSARY!!
        ###########################################################################################################

        # # TODO include if DDP is used
        # # Get features from all GPUs to increase negatives for contrastive loss.
        # # These will be lists of tensors with length = world size.
        # all_image_feats = dist.gather_across_processes(image_feats)
        # # shape: (batch_size * world_size, D), D=self.prototype_shape[1]
        #  all_image_feats = torch.cat(all_image_feats, dim=0)

        # Compute all necessary hyperbolic components. We enclose the entire block with
        # autocast to force a higher floating point precision.
        # with torch.autocast(self.device.type, dtype=torch.float32):
        #### Lorentz distance of each prototype from its corresponding extracted feature.  Shape (N, P)
        part_feat_prot_lorentz_distance = L.elementwise_dist(hyperbolic_feature_map,  # (N, P, D)
                                                             prototype_vectors,  # (P, D)
                                                             _curv)
        part_prototype_activations = self.distance_2_similarity(part_feat_prot_lorentz_distance)  # shape (N, P)

        #### Lorentz distance of local prototypes from local featurtes. shape (N, PL)
        #### Calculate lorentzian distance in an elementwise fashion, to save computation
        local_feat_prot_lorentz_distance = L.elementwise_dist(hyperbolic_local_feats,  # shape (N, PL, D)
                                                               local_prototype_vectors,  # shape (PL, D)
                                                               _curv)  # shape (N, PL)
        local_prototype_activations = self.distance_2_similarity(local_feat_prot_lorentz_distance)  # shape (N, PL)

        # classification layer
        if self.cls_method == "broad":
            logits = self.last_layer(part_prototype_activations)  # shape (N, num_classes)
        elif self.cls_method == "both":
            logits = self.last_layer(part_prototype_activations)  # shape (N, num_classes)
            logits_local = self.last_layer_local(local_prototype_activations)  # shape (N, num_classes)
            # logits = 0.5*(logits + logits_local)
            logits = (logits, logits_local)
        elif self.cls_method == "local":
            logits = self.last_layer_local(local_prototype_activations)  # shape (N, num_classes)
        # if self.cls_method == "broad":
        #     prototype_activations = part_prototype_activations
        # elif self.cls_method == "both":
        #     prototype_activations = torch.cat([part_prototype_activations, local_prototype_activations], dim=-1) # shape (N, P+PL)
        # elif self.cls_method == "local":
        #     prototype_activations = local_prototype_activations
        #     # logits = local_prototype_activations  # shape (N, num_classes)
        # logits = self.last_layer(prototype_activations)  # shape (N, num_classes)

        # TODO check what should be returned!
        if not self.lift_prototypes:
        #### if Option 1 is selected to not lift the prototypes, return the hyperbolic features!
            features_to_track = hyperbolic_feature_map  # shape (N, P, D)
            local_features_to_track = hyperbolic_local_feats   # shape (N, L, D)
        #### if Option 2 is selected to lift the prototypes, return the euclidean features!
        else:
            # TODO or try logmap0 of the hyperbolic_feature_map!
            features_to_track = features_extracted  # shape (N, P, D)
            local_features_to_track = local_feats_extracted   # shape (N, L, D)

        return (features_to_track, part_feat_prot_lorentz_distance, occurrence_map,
                local_features_to_track, local_feat_prot_lorentz_distance, local_attn_map, logits)

    def get_prototype_vectors(self):
        if self.lift_prototypes:
            prototype_vectors = L.get_hyperbolic_feats(self.prototype_vectors.squeeze(),
                                                       self.visual_alpha, self.curv, self.device)  # shape (P, D)
            local_prototype_vectors = L.get_hyperbolic_feats(self.local_prototype_vectors.squeeze(),
                                                              self.visual_alpha, self.curv, self.device)  # shape (P, D)
        else:
            prototype_vectors = self.prototype_vectors.squeeze()
            local_prototype_vectors = self.local_prototype_vectors.squeeze()

        return prototype_vectors, local_prototype_vectors

    def __repr__(self):
        # XProtoNet(self, backbone, img_size, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            "Hyperbolic_XProtoNet(\n"
            "\tcnn_backbone: {},\n"
            "\timg_size: {},\n"
            "\tnum_local_prototypes_per_class: {},\n"
            "\tprototype_shape: {},\n"
            "\tproto_layer_rf_info: {},\n"
            "\tnum_classes: {},\n"
            "\tepsilon: {}\n"
            ")"
        )

        return rep.format(
            self.cnn_backbone,
            self.img_size,
            self.num_local_prototypes_per_class,
            self.prototype_shape,
            self.proto_layer_rf_info,
            self.num_classes,
            self.epsilon,
        )

    def get_last_layer_prot_class_identity(self, layer_name="last_layer"):
        if layer_name == "last_layer":
            assert (self.cls_method == "broad") or (self.cls_method == "both")
            return self.prototype_class_identity
        if layer_name == "last_layer_local":
            assert (self.cls_method == "local") or (self.cls_method == "both")
            return self.local_prototype_class_identity

def construct_Hyper_XProtoNet(
    base_architecture,
    pretrained=True,
    img_size=224,
    prototype_shape=(2000, 512, 1, 1),
    num_classes=200,
    prototype_activation_function="linear",
    add_on_layers_type="bottleneck",
    feat_range_type="Sigmoid",
    num_local_prototypes_per_class=1,
    cls_method="broad",
    local_prototype_method="attn",
    local_feat_method="1_per_input",
    lift_prototypes=False,
    learn_curv=False,
    local_prot_donut=1.0,
    **kwargs
):
    cnn_backbone = base_architecture_to_features[base_architecture](pretrained=pretrained)
    # layer_filter_sizes, layer_strides, layer_paddings = cnn_backbone.conv_info()
    # proto_layer_rf_info = compute_proto_layer_rf_info_v2(
    #     img_size=img_size,
    #     layer_filter_sizes=layer_filter_sizes,
    #     layer_strides=layer_strides,
    #     layer_paddings=layer_paddings,
    #     prototype_kernel_size=prototype_shape[2],
    # )
    return Hyper_XProtoNet(
        cnn_backbone=cnn_backbone,
        img_size=img_size,
        prototype_shape=prototype_shape,
        proto_layer_rf_info=None,
        num_classes=num_classes,
        learn_curv=learn_curv,
        init_weights=True,
        prototype_activation_function=prototype_activation_function,
        add_on_layers_type=add_on_layers_type,
        feat_range_type=feat_range_type,
        num_local_prototypes_per_class=num_local_prototypes_per_class,
        cls_method=cls_method,
        local_prototype_method=local_prototype_method,
        local_feat_method=local_feat_method,
        lift_prototypes=lift_prototypes,
        local_prot_donut=local_prot_donut,
    )
