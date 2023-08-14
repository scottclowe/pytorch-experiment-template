import torch
from timm.data.transforms import _str_to_pil_interpolation
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

    if transform_type == "imagenet":
        interpolation = _str_to_pil_interpolation[args.get("interpolation", "bicubic")]
        crop_pct = args.get("crop_pct", 0.875)
        horz_flip = args.get("horz_flip", 0.5)

        train_transform = transforms.Compose(
            [
                transforms.Resize(int(image_size / crop_pct), interpolation),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=horz_flip),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize(int(image_size / crop_pct), interpolation),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )

    elif transform_type == "simple":
        padding = args.get("padding", 4)
        horz_flip = args.get("horz_flip", 0.5)

        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(image_size, padding=padding),
                transforms.RandomHorizontalFlip(p=horz_flip),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    elif transform_type == "scale-flip":
        horz_flip = args.get("horz_flip", 0.5)

        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=horz_flip),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    elif transform_type == "barebones":
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        raise NotImplementedError

    return (train_transform, test_transform)
