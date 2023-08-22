import warnings

import timm
import torch
from timm.data import resolve_data_config


def get_timm_encoder(model_name, pretrained=False, in_chans=3):
    r"""
    Get the encoder model and its configuration from timm.

    Parameters
    ----------
    model_name : str
        Name of the model to load.
    pretrained : bool, default=False
        Whether to load the model with pretrained weights.
    in_chans : int, default=3
        Number of input channels.

    Returns
    -------
    encoder : torch.nn.Module
        The encoder model (with pretrained weights loaded if requested).
    encoder_config : dict
        The data configuration of the encoder model.
    """
    if len(timm.list_models(model_name)) == 0:
        warnings.warn(
            f"Unrecognized model '{model_name}'. Trying to fetch it from the"
            " hugging-face hub.",
            UserWarning,
            stacklevel=2,
        )
        model_name = "hf-hub:timm/" + model_name

    # We request the model without the classification head (num_classes=0)
    # to get it as an encoder-only model
    encoder = timm.create_model(
        model_name, pretrained=pretrained, num_classes=0, in_chans=in_chans
    )
    encoder_config = resolve_data_config({}, model=encoder)

    # Send a dummy input through the encoder to find out the shape of its output
    encoder.eval()
    dummy_output = encoder(torch.zeros((1, *encoder_config["input_size"])))
    encoder_config["n_feature"] = dummy_output.shape[1]
    encoder.train()

    encoder_config["in_channels"] = encoder_config["input_size"][0]

    return encoder, encoder_config
