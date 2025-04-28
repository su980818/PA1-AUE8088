from torchmetrics import Metric
import torch
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # Initialize state for each class
        self.add_state("true_positives", default=torch.zeros(num_classes) ,dist_reduce_fx='sum')
        self.add_state("false_positives", default=torch.zeros(num_classes) ,dist_reduce_fx='sum')
        self.add_state("false_negatives", default=torch.zeros(num_classes) ,dist_reduce_fx='sum')
        #self.add_state("all_preds", default=[], dist_reduce_fx=None)
        #self.add_state("all_targets", default=[], dist_reduce_fx=None)

    def update(self, preds, target):
        # The preds (B x C tensor)
        # Convert logits to predicted class indices
        preds = torch.argmax(preds, dim=-1)
        #self.all_preds.extend(preds.cpu().tolist())
        #self.all_targets.extend(target.cpu().tolist())


        # Check that shapes match
        assert preds.shape == target.shape, f"ERROR: Shape miss match : {preds.shape} vs {target.shape}"

        # calculate true positives, false positives, and false negatives per class
        for c in range(self.num_classes):
            self.true_positives[c] += torch.sum((preds == c) & (target == c)).float()
            self.false_positives[c] += torch.sum((preds == c) & (target != c)).float()
            self.false_negatives[c] += torch.sum((preds != c) & (target == c)).float()

        
    def compute(self):
        #print(f1_score(self.all_targets, self.all_preds, average='macro'))
        #  Macro average 
        f1_scores = 2*self.true_positives / (2*self.true_positives + self.false_positives  + self.false_negatives + 1e-8)
        #print("My_f1:",f1_scores.mean(),'\n')
        return f1_scores.mean()




class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, dim=-1)

        # [TODO] check if preds and target have equal shape
        assert preds.shape == target.shape, f"ERROR: Shape miss match : {preds.shape} vs {target.shape}"

        # [TODO] Count the number of correct prediction
        correct = torch.sum(preds == target)

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()
   
    def compute(self):
        return self.correct.float() / self.total.float()


