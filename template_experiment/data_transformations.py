import timm.data
import torch
from torchvision import transforms

NORMALIZATION = {
    "imagenet": [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
    "mnist": [(0.1307,), (0.3081,)],
    "cifar": [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)],
}

VALID_TRANSFORMS = ["imagenet", "cifar", "mnist"]


def get_transform(transform_type="barebones", image_size=32, args=None):
    if args is None:
        args = {}
    mean, std = NORMALIZATION[args.get("normalization", "imagenet")]
    if "mean" in args:
        mean = args["mean"]
    if "std" in args:
        std = args["std"]

    if transform_type == "noaug":
        # No augmentations, just resize and normalize.
        # N.B. If the raw training image isn't square, there is a small
        # "augmentation" as we will randomly crop a square (of length equal to
        # the shortest side) from it. We do this because we assume inputs to
        # the network must be square.
        train_transform = transforms.Compose(
            [
                transforms.Resize(image_size),  # Resize shortest side to image_size
                transforms.RandomCrop(image_size),  # If it is not square, *random* crop
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(image_size),  # Resize shortest side to image_size
                transforms.CenterCrop(image_size),  # If it is not square, center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    elif transform_type == "imagenet":
        # Appropriate for really large natual images, as in ImageNet.
        # For training:
        # - Zoom in randomly with scale (big range of how much to zoom in)
        # - Stretch with random aspect ratio
        # - Flip horizontally
        # - Randomly adjust brightness/contrast/saturation
        # - (No rotation or skew)
        # - Interpolation is randomly either bicubic or bilinear
        train_transform = timm.data.create_transform(
            input_size=image_size,
            is_training=True,
            scale=(0.08, 1.0),  # default imagenet scale range
            ratio=(3.0 / 4.0, 4.0 / 3.0),  # default imagenet ratio range
            hflip=0.5,
            vflip=0.0,
            color_jitter=0.4,
            interpolation="random",
            mean=mean,
            std=std,
        )
        # For testing:
        # - Zoom in 87.5%
        # - Center crop
        # - Interpolation is bilinear
        test_transform = transforms.Compose(
            [
                transforms.Resize(int(image_size / 0.875)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )

    elif transform_type == "cifar":
        # Appropriate for smaller natural images, as in CIFAR-10/100.
        # For training:
        # - Zoom in randomly with scale (small range of how much to zoom in by)
        # - Stretch with random aspect ratio
        # - Flip horizontally
        # - Randomly adjust brightness/contrast/saturation
        # - (No rotation or skew)
        train_transform = timm.data.create_transform(
            input_size=image_size,
            is_training=True,
            scale=(0.7, 1.0),  # reduced scale range
            ratio=(3.0 / 4.0, 4.0 / 3.0),  # default imagenet ratio range
            hflip=0.5,
            vflip=0.0,
            color_jitter=0.4,  # default imagenet color jitter
            interpolation="random",
            mean=mean,
            std=std,
        )
        # For testing:
        # - Resize to desired size only, with a center crop step included in
        #   case the raw image was not square.
        test_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )

    elif transform_type == "digits":
        # Appropriate for smaller images containing digits, as in MNIST.
        # - Zoom in randomly with scale (small range of how much to zoom in by)
        # - Stretch with random aspect ratio
        # - Don't flip the images (that would change the digit)
        # - Randomly adjust brightness/contrast/saturation
        # - (No rotation or skew)
        train_transform = timm.data.create_transform(
            input_size=image_size,
            is_training=True,
            scale=(0.7, 1.0),  # reduced scale range
            ratio=(3.0 / 4.0, 4.0 / 3.0),  # default imagenet ratio range
            hflip=0.0,
            vflip=0.0,
            color_jitter=0.4,  # default imagenet color jitter
            interpolation="random",
            mean=mean,
            std=std,
        )
        # For testing:
        # - Resize to desired size only, with a center crop step included in
        #   case the raw image was not square.
        test_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )

    elif transform_type == "autoaugment-imagenet":
        # Augmentation policy learnt by AutoAugment, described in
        # https://arxiv.org/abs/1805.09501
        # The policies mostly concern changing the colours of the image,
        # but there is a little rotation and shear too. We need to include
        # our own random cropping, stretching, and flipping.
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
                ),
                transforms.AutoAugment(
                    policy=transforms.AutoAugmentPolicy.IMAGENET,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )
        # For testing:
        # - Zoom in 87.5%
        # - Center crop
        test_transform = transforms.Compose(
            [
                transforms.Resize(int(image_size / 0.875)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )

    elif transform_type == "autoaugment-cifar":
        # Augmentation policy learnt by AutoAugment, described in
        # https://arxiv.org/abs/1805.09501
        # The policies mostly concern changing the colours of the image,
        # but there is a little rotation and shear too. We need to include
        # our own random cropping, stretching, and flipping.
        train_transform = transforms.Compose(
            [
                transforms.AutoAugment(
                    policy=transforms.AutoAugmentPolicy.CIFAR10,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.RandomResizedCrop(
                    image_size, scale=(0.7, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
                ),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )
        # For testing:
        # - Resize to desired size only, with a center crop step included in
        #   case the raw image was not square.
        test_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )

    elif transform_type == "randaugment-imagenet":
        # Augmentation policy learnt by RandAugment, described in
        # https://arxiv.org/abs/1909.13719
        train_transform = transforms.Compose(
            [
                transforms.RandAugment(
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.RandomResizedCrop(
                    image_size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
                ),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )
        # For testing:
        # - Zoom in 87.5%
        # - Center crop
        test_transform = transforms.Compose(
            [
                transforms.Resize(int(image_size / 0.875)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )

    elif transform_type == "randaugment-cifar":
        # Augmentation policy learnt by RandAugment, described in
        # https://arxiv.org/abs/1909.13719
        train_transform = transforms.Compose(
            [
                transforms.RandAugment(
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.RandomResizedCrop(
                    image_size, scale=(0.7, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
                ),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )
        # For testing:
        # - Resize to desired size only, with a center crop step included in
        #   case the raw image was not square.
        test_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )

    elif transform_type == "trivialaugment-imagenet":
        # Trivial augmentation policy, described in https://arxiv.org/abs/2103.10158
        train_transform = transforms.Compose(
            [
                transforms.TrivialAugmentWide(
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.RandomResizedCrop(
                    image_size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
                ),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )
        # For testing:
        # - Zoom in 87.5%
        # - Center crop
        test_transform = transforms.Compose(
            [
                transforms.Resize(int(image_size / 0.875)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )

    elif transform_type == "trivialaugment-cifar":
        # Trivial augmentation policy, described in https://arxiv.org/abs/2103.10158
        train_transform = transforms.Compose(
            [
                transforms.TrivialAugmentWide(
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.RandomResizedCrop(
                    image_size, scale=(0.7, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
                ),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )
        # For testing:
        # - Resize to desired size only, with a center crop step included in
        #   case the raw image was not square.
        test_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )

    else:
        raise NotImplementedError

    return (train_transform, test_transform)
