import torch
from opengait.modeling.base_model import BaseModel
from opengait.modeling.modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks
from einops import rearrange
import numpy as np

class ScoNet(BaseModel):

    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

    def forward(self, inputs):
        ipts, pids, labels, _, seqL = inputs

        # Custom Label mapping for 2 classes
        # 'healthy'/'negative'/'neutral' -> 0
        # 'patient'/'positive' -> 1
        # If labels are already integers (0/1), use them directly.
        
        # Identify if labels are strings or ints
        if len(labels) > 0 and isinstance(labels[0], str):
            label_map = {
                'healthy': 0, 'negative': 0, 'neutral': 0,  # Mapping neutral to healthy
                'patient': 1, 'positive': 1
            }
            # Fallback for unforeseen labels to avoid crash, mapping to 0 (healthy) by default
            # label_ids = np.array([label_map.get(status.lower(), 0) for status in labels])
            
            # SUBSTRING MATCHING for "patient_00", "healthy_01" style labels
            ids = []
            for status in labels:
                s = status.lower()
                if 'patient' in s or 'positive' in s:
                    ids.append(1)
                else:
                    # 'healthy', 'negative', 'neutral' or anything else -> 0
                    ids.append(0)
            label_ids = np.array(ids)
        else:
            # Assume already processed int labels
            label_ids = np.array(labels)

        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        else:
            sils = rearrange(sils, 'n s c h w -> n c s h w')

        del ipts
        outs = self.Backbone(sils)  # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
        embed = embed_1
        
        # Logits shape match check (debug helper)
        # Expected: [batch, class_num, parts]
        
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': pids},
                'softmax': {'logits': logits, 'labels': label_ids},
            },
            'visual_summary': {
                'image/sils': rearrange(sils,'n c s h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': logits
            }
        }
        return retval
