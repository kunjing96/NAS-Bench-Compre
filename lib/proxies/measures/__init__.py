import os
import glob


__PROXY_DICT = {}


def available_proxy():
    return __PROXY_DICT.keys()


def measure(name, bn=True, copy_net=True, force_clean=True, **impl_args):
    def make_impl(func):
        def measure_impl(net_orig, device, *args, **kwargs):
            if copy_net:
                net = net_orig.get_prunable_copy(bn=bn).to(device)
            else:
                net = net_orig
            ret = func(net, device, *args, **kwargs, **impl_args)
            if copy_net and force_clean:
                import gc
                import torch
                del net
                torch.cuda.empty_cache()
                gc.collect()
            return ret

        if name in __PROXY_DICT:
            raise KeyError(f'Duplicated measure! {name}')
        __PROXY_DICT.update({name: measure_impl})
        return func
    return make_impl


def calc_measure(name, net, device, *args, **kwargs):
    return __PROXY_DICT[name](net, device, *args, **kwargs)


__import__(name="lib.proxies." + os.path.basename(os.path.dirname(__file__)), fromlist=[os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(os.path.dirname(__file__), "[!_]*.py"))])
