
# ensure that lr_g and lr_d are set if there's either of them set
# or learning_rate is set (it is then used for both)
def lr_check(p:dict):

    if "lr_g" not in p and "lr_d" not in p:
        if "learning_rate" in p:
            p["lr_g"] = p["learning_rate"]
            p["lr_d"] = p["learning_rate"]
    if "lr_g" not in p and "lr_d" in p:
        p["lr_g"] = p["lr_d"]
    if "lr_d" not in p and "lr_g" in p:
        p["lr_d"] = p["lr_g"]

# TODO : when there is support for first upscale/downscale =/= 4, check it here
def structure_check(p:dict):
    if not p.keys() >= {"features_d", "features_g", "upscales", "downscales"}:
        s, fg, fd = doubling_arch_builder(p["img_size"], p["features"])
        p["features_g"] = fg
        p["features_d"] = fd
        p["upscales"] = s
        p["downscales"] = s
    else:
        from math import prod
        img_size = p["img_size"]
        assert prod(p["upscales"])*4 == img_size, "Upscales list doesn't produce image_size"
        assert prod(p["downscales"])*4 == img_size, "Downscales list doesn't reduce image_size"

def disc_features(
        n_layers: int, 
        base_features: int, 
        features_list: "list[int]" = None
    ) -> "list[int]":
    """Derive list of features for generator tail layers.

    Args:
        n_layers (int): Amount of tail layers.
        base_features (int): Base amount of features. 
            (i.e. in_c of the first mid layer)
        feature_list (list[int], optional): List of in_c for the tail layers. 
            If missing, every layer will have double the preceding layer's features. 
            Defaults to None.
    """
    
    if not features_list:
        features_list = [base_features]
    assert features_list[0] == base_features, "Bogus features list."
    adds = [features_list[-1] * 2**(i+1) for i in range(n_layers - len(features_list))]
    return features_list + adds

def gen_features(
        n_layers: int, 
        base_features: int, 
        features_list: "list[int]" = None
    ) -> "list[int]":
    return disc_features(n_layers, base_features, features_list)[::-1]

def doubling_arch_builder(img_size: int, base_features: int):
    """Builds required parameters for an architectures that does x2 upscales/downscales
    and doubles their layer feature, based on an image size and base number of features.

    Args:
        img_size (int): Image size for the models. (i.e. dataset and generator output)
        base_features (int): Base amount of conv layer features.

    Returns:
        Parameters (Tuple[int, int, int]): Upscale/Downscale factor list and features.
    """
    scalings = []
    size = 4
    while size < img_size:
        scalings.append(2)
        size *= 2
    features_d = disc_features(len(scalings), base_features)
    features_g = features_d[::-1]
    return scalings, features_d, features_g
    