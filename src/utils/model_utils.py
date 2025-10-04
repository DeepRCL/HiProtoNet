import torch


def get_prototype_class_identity(num_prototypes, num_classes):
    """
    Here we are initializing the class identities of the local prototypes
    """
    assert num_prototypes % num_classes == 0
    # a onehot indication matrix for each ghlobal prototype's class identity
    prototype_class_identity = torch.zeros(num_prototypes, num_classes)

    num_prototypes_per_class = num_prototypes // num_classes
    for j in range(num_prototypes):
        prototype_class_identity[j, j // num_prototypes_per_class] = 1

    return prototype_class_identity