import torch


def motion_augmentation(x: torch.Tensor, n: int, shuffle: bool = False) -> torch.Tensor:
    """
    Performs motion augmentation on input tensor. 

    This method splits the input tensors into two parts based on the specified number of augmentations. 
    The first part (up to 'n' elements) is used for augmentation. In the case of the dynamic 
    input tensor, it typically undergoes random shuffling. The augmented parts are then concatenated back with the 
    non-augmented parts. 

    :param x: Input motion torch.Tensor [batch_size, feature_size]
    :param n: Integer, n first elements to be augmented
    :shuffle: Bool, it should undergo random shuffling
    :returns: Augmented torch.Tensor

    Note:
    - Augmentation involves detaching from the computation graph, which affects backpropagation.
    """
    batch_size = x.size(0)
    n = min(n, batch_size)
    x_aug = x[:n]
    x_rest = x[n:]  
    if shuffle:
        indices = torch.randperm(n)
        x_aug = x_aug[indices]
    x_aug = x_aug.detach()  
    x_final = torch.cat([x_aug, x_rest], dim=0)

    return x_final
