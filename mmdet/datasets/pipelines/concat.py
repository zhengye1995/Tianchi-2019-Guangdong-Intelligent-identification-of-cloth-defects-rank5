from ..registry import PIPELINES
import mmcv
import numpy as np


@PIPELINES.register_module
class Concat(object):
    """Concat two image.

    Args:
        template_path: template images path
    """

    def __init__(self, template_path):
        self.template_path = template_path

    def __call__(self, results):
        if 'concat_img' not in results or results['concat_img'] is None:
            template_name = 'template_' + results['img_info']['filename'].split('_')[0] + '.jpg'
            template_im_name = self.template_path + results['img_info']['filename'].split('.')[0] + '/' + template_name
            img_temp = mmcv.imread(template_im_name)
            results['img'] = np.concatenate([results['img'], img_temp], axis=2)
            results['concat'] = True
        else:
            results['img'] = np.concatenate([results['img'], results['concat_img']], axis=2)
            results['concat'] = True
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(template_path={})'.format(
            self.template_path)
        return repr_str