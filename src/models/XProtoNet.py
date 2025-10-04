import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.receptive_field import compute_proto_layer_rf_info_v2
from src.models.ProtoPNet import PPNet, base_architecture_to_features


class XProtoNet(PPNet):
    def __init__(self, init_weights=True, **kwargs):
        super(XProtoNet, self).__init__(init_weights=False, **kwargs)

        cnn_backbone_out_channels = self.get_cnn_backbone_out_channels(self.cnn_backbone)

        # feature extractor module. We remove the sigmoid layer
        # self.add_on_layers = torch.nn.Sequential(*list(self.add_on_layers.children())[:-1])

        # Occurrence map module
        self.occurrence_module = nn.Sequential(
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
                out_channels=self.prototype_shape[0],
                kernel_size=1,
                bias=False,
            ),
        )

        # Last classification layer, redefine to initialize randomly
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)  # do not use bias

        self.om_softmax = nn.Softmax(dim=-1)
        self.cosine_similarity = nn.CosineSimilarity(dim=2)

        # TODO maybe separate last layer into heads?

        if init_weights:
            self._initialize_weights(self.add_on_layers)
            self._initialize_weights(self.occurrence_module)
            self.set_last_layer_incorrect_connection(layer_name="last_layer", incorrect_strength=0)

    def forward(self, x):
        # Feature Extractor Layer
        x = self.cnn_backbone(x)
        feature_map = self.add_on_layers(x).unsqueeze(1)  # shape (N, 1, 128, H, W)
        occurrence_map = self.get_occurence_map(x)  # shape (N, P, 1, H, W)
        features_extracted = (occurrence_map * feature_map).sum(dim=3).sum(dim=3)  # shape (N, P, 128)
        # # TODO need to do average! not sum! eh?
        # features_extracted = (occurrence_map * feature_map).mean(dim=(3, 4))  # shape (N, P, 128)

        # Prototype Layer
        similarity = self.cosine_similarity(
            features_extracted, self.prototype_vectors.squeeze().unsqueeze(0)
        )  # shape (N, P)
        similarity = (similarity + 1) / 2.0 # normalizing to [0,1] for positive reasoning  # TODO modify in future

        # classification layer
        logits = self.last_layer(similarity)  #TODO check the squared weight idea

        return logits, similarity, occurrence_map

    def extract_features(self, x):
        # Feature Extractor Layer
        x = self.cnn_backbone(x)
        feature_map = self.add_on_layers(x).unsqueeze(1)  # shape (N, 1, 128, H, W)
        occurrence_map = self.get_occurence_map(x)  # shape (N, P, 1, H, W)
        features_extracted = (occurrence_map * feature_map).sum(dim=3).sum(dim=3)  # shape (N, P, 128)
        # # TODO need to do average! not sum! eh?
        # features_extracted = (occurrence_map * feature_map).mean(dim=(3, 4))  # shape (N, P, 128)

        # Prototype Layer
        similarity = self.cosine_similarity(
            features_extracted, self.prototype_vectors.squeeze().unsqueeze(0)
        )  # shape (N, P)
        similarity = (similarity + 1) / 2.0  # normalizing to [0,1] for positive reasoning

        # classification layer
        logits = self.last_layer(similarity)

        # shape (N, C)       (N, P, D)      (N, P)
        return logits, features_extracted, similarity

    def compute_occurence_map(self, x):
        # Feature Extractor Layer
        x = self.cnn_backbone(x)
        occurrence_map = self.get_occurence_map(x)  # shape (N, P, 1, H, W)
        return occurrence_map

    def get_occurence_map_softmaxed(self, x):
        occurrence_map = self.occurrence_module(x)  # shape (N, P, H, W)
        n, p, h, w = occurrence_map.shape
        occurrence_map = occurrence_map.reshape((n, p, -1))
        occurrence_map = self.om_softmax(occurrence_map).reshape((n, p, h, w)).unsqueeze(2)  # shape (N, P, 1, H, W)
        return occurrence_map

    def get_occurence_map_absolute_val(self, x):
        occurrence_map = self.occurrence_module(x)  # shape (N, P, H, W)
        occurrence_map = torch.abs(occurrence_map).unsqueeze(2)  # shape (N, P, 1, H, W)
        return occurrence_map

    def get_occurence_map_sigmoid_norm(self, x):
        occurrence_map = self.occurrence_module(x)  # shape (N, P, H, W)
        occurrence_map = F.sigmoid(occurrence_map).unsqueeze(2)  # shape (N, L, 1, H, W)
        return occurrence_map

    def get_occurence_map_min_max_norm(self, x):
        occurrence_map = self.occurrence_module(x)  # shape (N, P, H, W)

        # Option 3: min-max norm: subtract min, divide by max! to be in range [0-1]
        occurrence_map = occurrence_map - occurrence_map.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        epsilon = 1e-8
        occurrence_map = occurrence_map / (epsilon + occurrence_map.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0])
        occurrence_map = occurrence_map / (epsilon + occurrence_map.sum(dim=(2, 3), keepdim=True))
        occurrence_map = occurrence_map.unsqueeze(2)  # shape (N, L, 1, H, W)

        return occurrence_map


    def get_occurence_map(self, x):
        # Option 1: Absolute value
        # occurrence_map = self.get_occurence_map_absolute_val(x)  # shape (N, P, 1, H, W)
        # Option 2: Softmax!
        # occurrence_map = self.get_occurence_map_softmaxed(x)  # shape (N, P, 1, H, W)
        # Option 3: min-max norm: subtract min, divide by max! to be in range [0-1]
        occurrence_map = self.get_occurence_map_min_max_norm(x)  # shape (N, P, 1, H, W)
        # Option 4: Sigmoid like Xprotonet
        # occurrence_map = self.get_occurence_map_sigmoid_norm(x)  # shape (N, P, 1, H, W)
        return occurrence_map

    def push_forward(self, x):
        """
        this method is needed for the pushing operation
        """
        # Feature Extractor Layer
        x = self.cnn_backbone(x)
        feature_map = self.add_on_layers(x).unsqueeze(1)  # shape (N, 1, 128, H, W)
        occurrence_map = self.get_occurence_map(x)  # shape (N, P, 1, H, W)
        features_extracted = (occurrence_map * feature_map).sum(dim=3).sum(dim=3)  # shape (N, P, 128)
        # # TODO need to do average! not sum! eh?
        # features_extracted = (occurrence_map * feature_map).mean(dim=(3, 4))  # shape (N, P, 128)

        # Prototype Layer
        similarity = self.cosine_similarity(
            features_extracted, self.prototype_vectors.squeeze().unsqueeze(0)
        )  # shape (N, P)
        similarity = (similarity + 1) / 2.0  # normalizing to [0,1] for positive reasoning

        # classification layer
        logits = self.last_layer(similarity)

        return features_extracted, 1 - similarity, occurrence_map, logits

    def __repr__(self):
        # XProtoNet(self, backbone, img_size, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            "PPNet(\n"
            "\tcnn_backbone: {},\n"
            "\timg_size: {},\n"
            "\tprototype_shape: {},\n"
            "\tproto_layer_rf_info: {},\n"
            "\tnum_classes: {},\n"
            "\tepsilon: {}\n"
            ")"
        )

        return rep.format(
            self.cnn_backbone,
            self.img_size,
            self.prototype_shape,
            self.proto_layer_rf_info,
            self.num_classes,
            self.epsilon,
        )


def construct_XProtoNet(
    base_architecture,
    pretrained=True,
    img_size=224,
    prototype_shape=(2000, 512, 1, 1),
    num_classes=200,
    prototype_activation_function="log",
    add_on_layers_type="bottleneck",
    feat_range_type="Sigmoid",
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
    return XProtoNet(
        cnn_backbone=cnn_backbone,
        img_size=img_size,
        prototype_shape=prototype_shape,
        proto_layer_rf_info=None,
        num_classes=num_classes,
        init_weights=True,
        prototype_activation_function=prototype_activation_function,
        add_on_layers_type=add_on_layers_type,
        feat_range_type=feat_range_type,
    )
