import json
import torch

from lib.download import download_url
from lib.nets.OFA.networks import get_net_by_name
from lib.nets.OFA.elastic_nn.networks import OFAMobileNetV3, OFAProxylessNASNets, OFAResNets


def ofa_specialized(net_id, pretrained=True):
    url_base = "https://hanlab.mit.edu/files/OnceForAll/ofa_specialized/"
    net_config = json.load(
        open(
            download_url(
                url_base + net_id + "/net.config",
                model_dir=".torch/ofa_specialized/%s/" % net_id,
            )
        )
    )
    net = get_net_by_name(net_config["name"]).build_from_config(net_config)

    image_size = json.load(
        open(
            download_url(
                url_base + net_id + "/run.config",
                model_dir=".torch/ofa_specialized/%s/" % net_id,
            )
        )
    )["image_size"]

    if pretrained:
        init = torch.load(
            download_url(
                url_base + net_id + "/init",
                model_dir=".torch/ofa_specialized/%s/" % net_id,
            ),
            map_location="cpu",
        )["state_dict"]
        net.load_state_dict(init)
    return net, image_size


def ofa_net(net_id, pretrained=True):
    if net_id == "ofa_proxyless_d234_e346_k357_w1.3":
        net = OFAProxylessNASNets(
            dropout_rate=0,
            width_mult=1.3,
            ks_list=[3, 5, 7],
            expand_ratio_list=[3, 4, 6],
            depth_list=[2, 3, 4],
        )
    elif net_id == "ofa_mbv3_d234_e346_k357_w1.0":
        net = OFAMobileNetV3(
            dropout_rate=0,
            width_mult=1.0,
            ks_list=[3, 5, 7],
            expand_ratio_list=[3, 4, 6],
            depth_list=[2, 3, 4],
        )
    elif net_id == "ofa_mbv3_d234_e346_k357_w1.2":
        net = OFAMobileNetV3(
            dropout_rate=0,
            width_mult=1.2,
            ks_list=[3, 5, 7],
            expand_ratio_list=[3, 4, 6],
            depth_list=[2, 3, 4],
        )
    elif net_id == "ofa_resnet50":
        net = OFAResNets(
            dropout_rate=0,
            depth_list=[0, 1, 2],
            expand_ratio_list=[0.2, 0.25, 0.35],
            width_mult_list=[0.65, 0.8, 1.0],
        )
        net_id = "ofa_resnet50_d=0+1+2_e=0.2+0.25+0.35_w=0.65+0.8+1.0"
    else:
        raise ValueError("Not supported: %s" % net_id)

    if pretrained:
        url_base = "https://hanlab.mit.edu/files/OnceForAll/ofa_nets/"
        init = torch.load(
            download_url(url_base + net_id, model_dir=".torch/ofa_nets"),
            map_location="cpu",
        )["state_dict"]
        net.load_state_dict(init)
    return net
