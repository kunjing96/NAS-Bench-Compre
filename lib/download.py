import os
import sys

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


def download_url(url, model_dir="~/.torch/", overwrite=False):
    target_dir = url.split("/")[-1]
    model_dir = os.path.expanduser(model_dir)
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_dir = os.path.join(model_dir, target_dir)
        cached_file = model_dir
        if not os.path.exists(cached_file) or overwrite:
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)
        return cached_file
    except Exception as e:
        # remove lock file so download can be executed next time.
        os.remove(os.path.join(model_dir, "download.lock"))
        sys.stderr.write("Failed to download from url %s" % url + "\n" + str(e) + "\n")
        return None
