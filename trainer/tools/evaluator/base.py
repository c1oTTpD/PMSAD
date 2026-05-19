import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from lib.utils.tools.logger import Logger as Log
from lib.metrics import running_score as rslib
from lib.metrics import F1_running_score as fscore_rslib
from lib.utils.distributed import get_world_size, get_rank, is_distributed


class _BaseEvaluator:
    

    def __init__(self, configer, trainer):
        self.configer = configer
        self.trainer = trainer
        self._init_running_scores()
        self.conditions = configer.conditions

    def use_me(self):
        raise NotImplementedError

    def _init_running_scores(self):
        raise NotImplementedError

    def update_score(self, *args, **kwargs):
        raise NotImplementedError
    
    def print_scores(self, show_miou=True):
        for key, rs in self.running_scores.items():
            Log.info('\nResult for {}'.format(key))
            if isinstance(rs, fscore_rslib.F1RunningScore):
                FScore, FScore_cls = rs.get_scores()
                Log.info('Mean FScore: {}'.format(FScore))
                Log.info(
                    'Class-wise FScore: {}'.format(
                        ', '.join(
                            '{:.3f}'.format(x)
                            for x in FScore_cls
                        )
                    )
                )
            elif isinstance(rs, rslib.SimpleCounterRunningScore):
                Log.info('ACC: {}\n'.format(rs.get_mean_acc()))
            else:
                if show_miou and hasattr(rs, 'get_mean_iou'):
                    mean_iou = rs.get_mean_iou()
                    cls_iu   = rs.get_cls_iou()
                    mean_jaccard_score   = rs.get_mean_jaccard_score()
                    acc = rs.get_pixel_acc()

                    rs.update_best_scores(mean_iou, cls_iu)
                    rs.update_jaccard_scores(mean_jaccard_score)
                    rs.update_acc(acc)

                    Log.info('Mean IOU: {}'.format(rs.get_mean_iou()))

                    # 输出每一类的IoU
                    Log.info('Class-wise IOU:')
                    for cls, iu in cls_iu.items():
                        Log.info('Class {}: {:.3f}'.format(cls, iu))
                    
                Log.info('Pixel ACC: {}'.format(rs.get_pixel_acc()))
                Log.info('Mean Jaccard: {}\n'.format(rs.get_mean_jaccard_score()))

                if hasattr(rs, 'n_classes') and rs.n_classes == 2:
                    Log.info(
                        'F1 Score: {} Precision: {} Recall: {}\n'
                        .format(*rs.get_F1_score())
                    )

        if hasattr(rs, 'best_mean_iou') and rs.best_mean_iou != 0:
            Log.info('Best Mean IOU: {}'.format(rs.best_mean_iou))
            Log.info('Best Class-wise IOU:')
            for cls, iu in rs.best_cls_iu.items():
                Log.info('Class {}: {:.3f}'.format(cls, iu))
            

        if hasattr(rs, 'best_jaccard_score') and rs.best_jaccard_score != 0:
            Log.info('Best Acc: {}'.format(rs.best_acc))
            Log.info('Best Mean Jaccard: {}\n'.format(rs.best_jaccard_score))
    

    def prepare_validaton(self):
        """
        Replicate models if using diverse size validation.
        """
        if is_distributed():
            return
        device_ids = list(range(len(self.configer.get('gpu'))))
        if self.conditions.diverse_size:
            cudnn.benchmark = False
            assert self.configer.get('val', 'batch_size') <= len(device_ids)
            replicas = nn.parallel.replicate(
                self.trainer.seg_net.module, device_ids)
            return replicas

    def update_performance(self):

        try:
            rs = self.running_scores[self.save_net_main_key]
            #if self.save_net_metric == 'miou':
                #perf = rs.get_mean_iou()
            if self.save_net_metric == 'miou':
                perf = rs.get_mean_jaccard_score()
            elif self.save_net_metric == 'acc':
                perf = rs.get_pixel_acc()

            max_perf = self.configer.get('max_performance')
            self.configer.update(['performance'], perf)
            if perf > max_perf and (not is_distributed() or get_rank() == 0):
                Log.info('Performance {} -> {}'.format(max_perf, perf))
        except Exception as e:
            Log.warn(e)

    def reset(self):
        for rs in self.running_scores.values():
            rs.reset()

    def reduce_scores(self):
        for rs in self.running_scores.values():
            if hasattr(rs, 'reduce_scores'):
                rs.reduce_scores()
