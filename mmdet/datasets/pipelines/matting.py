from ..registry import PIPELINES
import mmcv
import numpy as np
import cv2
from glob import glob


@PIPELINES.register_module
class Matting(object):
    """Matting bbox mask to new template image.

    Args:
        template_path: template images path
    """

    def __init__(self, template_path, ratio):
        self.template_path = template_path
        self.matting_ratio = ratio
        self.template_list = np.array(glob(template_path + '*/*.jpg'))

    def __call__(self, results):
        matting = True if np.random.rand() < self.matting_ratio else False
        if matting:
            template_no = np.random.randint(len(self.template_list))
            template_im_name = self.template_list[template_no]
            img_temp = mmcv.imread(template_im_name)
            img_temp = mmcv.imresize_like(img_temp, results['img'])
            results['concat_img'] = img_temp
            for bbox in results['gt_bboxes']:
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2])
                ymax = int(bbox[3])
                beta = np.random.uniform(0.5, 0.8)
                img_temp[ymin: ymax, xmin: xmax, :] = cv2.addWeighted(results['img'][ymin: ymax, xmin: xmax, :], beta,
                                                                      img_temp[ymin: ymax, xmin: xmax, :], 1-beta, 1)
            results['img'] = img_temp
        else:
            results['concat_img'] = None
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(template_path={})'.format(
            self.template_path)
        return repr_str