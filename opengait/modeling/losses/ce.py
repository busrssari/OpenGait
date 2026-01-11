import torch      
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import BaseLoss

class CrossEntropyLoss(BaseLoss):
    def __init__(self, scale=2**4, label_smooth=True, eps=0.1, loss_term_weight=1.0, log_accuracy=False, class_weights=None):
        super(CrossEntropyLoss, self).__init__(loss_term_weight)
        self.scale = scale
        self.label_smooth = label_smooth
        self.eps = eps
        self.log_accuracy = log_accuracy
        self.class_weights = class_weights

    def forward(self, logits, labels):
        """
            logits: [n, c, p]
            labels: [n]
        """
        n, c, p = logits.size()
        logits = logits.float()

         # ----- Minimal tip uyumu -----
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
           # print(f"Uyarı: labels NumPy array idi, tensor’a çevrildi. Şekil: {labels.shape}")

        labels = labels.to(logits.device)
        labels = labels.unsqueeze(1)  # orijinal kodda ne vardıysa onu bırakıyoruz


        # [CUSTOM ADDITION START] Class Imbalance Handling via Config
        weight = None
        if self.class_weights is not None and c == len(self.class_weights):
            weight = torch.tensor(self.class_weights).float().to(logits.device)
        # [CUSTOM ADDITION END]

        if self.label_smooth:
            loss = F.cross_entropy(
                logits*self.scale, labels.repeat(1, p), label_smoothing=self.eps, weight=weight)
        else:
            loss = F.cross_entropy(logits*self.scale, labels.repeat(1, p), weight=weight)
        
        self.info.update({'loss': loss.detach().clone()})
        if self.log_accuracy:
            pred = logits.argmax(dim=1)  # [n, p]
            accu = (pred == labels).float().mean()
            self.info.update({'accuracy': accu})
        return loss, self.info
