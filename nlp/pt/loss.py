import torch
import sklearn
from typing import Iterable


class MTCategoricalCrossEntropy(torch.nn.Module):
    """
    Loss module for multitask learning.
    It is tailored to multiclass classification, so it uses CrossEntropyLoss.
    The weights assigned to each class depend on trainable parameters.
    """
    def __init__(self,
                 intent_weight: Iterable[float] = None):
        """
        Args:
            intent_weight (Iterable[float], optional): List of class weights for intent classification. Defaults to None.
        """
        super(MTCategoricalCrossEntropy, self).__init__()
        self.log_vars = torch.nn.Parameter(torch.zeros((2)))
        self.ce1 = torch.nn.CrossEntropyLoss(weight=intent_weight, ignore_index=-100, reduction='none')
        self.ce2 = torch.nn.CrossEntropyLoss(weight=None, ignore_index=-100, reduction='none')

    def forward(self, predictions, targets):
        """
        Calculates the loss.
        Every task has a trainable parameter which adjusts the weight of the loss.

        Args:
            predictions (List[torch.Tensor]): List with model predictions. Each position is the prediction of a particular task.
                                              Shape is (N, n_intents), (N, n_tokens, n_entities).
            targets (List[torch.Tensor]): List with ground truth. Each position is the prediction of a particular task.
                                          Shape is (N,), (N, n_tokens).

        Returns:
            torch.Tensor: the loss, with no reduction, and shape (N,)
        """

        # Intent Classifier Loss
        loss1 = torch.exp(-self.log_vars[0]) * self.ce1(predictions[0], targets[0]) + self.log_vars[0]

        # NER Loss
        # We have to permute axis 1 and 2.
        # CCE works on dimension 1: (N, C, d1, d2, ..., dn). So we have to put the target dimension on position 1
        pred = predictions[1].permute(0, 2, 1)
        loss2 = torch.exp(-self.log_vars[1]) * self.ce2(pred, targets[1]) + self.log_vars[1]
        loss2 = torch.mean(loss2, axis=1)

        loss = loss1 + loss2

        return loss


class MTCrossEntropyLoss(torch.nn.Module):
    """
    Loss module for multitask learning.
    It is tailored to multiclass classification, so it uses CrossEntropyLoss.
    The weights assigned to each class depend on trainable parameters
    """
    def __init__(self,
                 intent_weight: Iterable[float] = None):
        """
        Args:
            intent_weight (Iterable[float], optional): List of class weights for intent classification. Defaults to None.
        """
        super(MTCrossEntropyLoss, self).__init__()
        self.log_vars = torch.nn.Parameter(torch.zeros((1)))
        self.ce1 = torch.nn.CrossEntropyLoss(weight=intent_weight, ignore_index=-100, reduction='none')
        self.ce2 = torch.nn.CrossEntropyLoss(weight=None, ignore_index=-100, reduction='none')

    def forward(self, predictions, targets):
        """
        Calculates the loss.
        Every task has a trainable parameter which adjusts the weight of the loss.

        Args:
            predictions (List[torch.Tensor]): List with model predictions. Each position is the prediction of a particular task.
                                              Shape is (N, n_intents), (N, n_tokens, n_entities)
            targets (List[torch.Tensor]): List with ground truth. Each position is the prediction of a particular task.
                                          Shape is (N,), (N, n_tokens)

        Returns:
            torch.Tensor: the loss, with no reduction, and shape (N,)
        """

        # Calculate weight for loss 1. Loss 2 weight is 1-loss1
        weight = torch.sigmoid(self.log_vars)[0]

        # Intent Classifier Loss
        loss1 = weight * self.ce1(predictions[0], targets[0])

        # NER Loss
        # We have to permute axis 1 and 2.
        # CCE works on dimension 1: (N, C, d1, d2, ..., dn). So we have to put the target dimension on position 1
        pred = predictions[1].permute(0, 2, 1)
        loss2 = (1 - weight) * self.ce2(pred, targets[1])
        loss2 = torch.mean(loss2, axis=1)

        loss = loss1 + loss2

        return loss


class LabelSmoothingCrossEntropyLoss(torch.nn.Module):
    """
    Categorical Cross Entropy in Keras style. If label smoothing is 0, then it works like torch.nn.CrossEntropyLoss.
    Equivalent to tf.keras.losses.CategoricalCrossEntropy.
    Adapted to receive one-hot tensors as the true label.
    In order to get the cross entropy loss, values must follow the pipeline:
    logits --> probs --> log --> loss
    Values can come in two versions from the model: logits (raw values) or
    probs (all together adding 1).
    This class includes label_smoothing feature
    """
    def __init__(self,
                 from_logits: bool = True,
                 label_smoothing: float = 0.0,
                 reduction: str = 'sum',
                 n_classes: int = None):
        """
        Args:
            from_logits (bool, optional): [description]. Defaults to False.
            label_smoothing (float, optional): [description]. Defaults to 0.0.
            reduction (str, optional): [description]. Defaults to 'sum'.
            n_classes (int, optional): Number of classes. Required only if ground truth is not one hot
        """
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.n_classes = n_classes

    def forward(self, y_pred, y_true):
        """
        Calculates the loss.

        Args:
            predictions (List[torch.Tensor]): List with model predictions. Each position is the prediction of a particular task.
                                              Shape is (N, n_intents), (N, n_tokens, n_entities)
            targets (List[torch.Tensor]): List with ground truth. Each position is the prediction of a particular task.
                                          Shape is (N,), (N, n_tokens)

        Returns:
            torch.Tensor: The loss. If there is any reduction, the shape is (N,).
                          If there is no reduction, shape is (N, n_classes)
        """
        # Adapt ground truth to one hot
        if len(y_true.shape) != len(y_pred.shape):
            y_true = torch.nn.functional.one_hot(y_true, num_classes=self.n_classes).float()

        # Label smoothing
        if self.label_smoothing > 0:
            y_true = y_true.clone()
            bias = self.label_smoothing/(y_true.shape[1]-1)
            slope = (1-self.label_smoothing)-bias
            y_true = slope*y_true+bias

        # Check if it comes from logits (no softmax applied) or the softmax needs to be applied
        # The output of this is Negative LogSoftmax
        if self.from_logits:
            y_pred = -torch.nn.functional.log_softmax(y_pred, dim=-1)
        else:
            y_pred = -torch.log(y_pred)

        # Calculate NLL loss
        loss = torch.sum(y_true * y_pred, dim=1)

        # Applies reduction method
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class LabelSmoothingMultiLabelCrossEntropyLoss(torch.nn.Module):
    """
    Adaptation of Categorical Crossentropy loss function with label smoothing for multilabel settings.
    Adapted to receive one-hot tensors as the true label.
    In order to get the cross entropy loss, values must follow the pipeline:
    logits --> individual probs --> log --> loss
    Values can come in two versions from the model: logits (raw values) or
    probs (sigmoid applied).
    This class includes label_smoothing feature
    """
    def __init__(self,
                 from_logits:bool = True,
                 label_smoothing:float = 0.0,
                 keep_sum:bool = True,
                 reduction:str = 'sum',
                 n_classes:int = None):
        """
        Args:
            from_logits (bool, optional): [description]. Defaults to False.
            label_smoothing (float, optional): [description]. Defaults to 0.0.
            keep_sum (bool, optional): Whether the smooth transformation maintains
                                       the sum of the row equal through the process
                                       by assuming internal synergy of data, or
                                       simply see each logit as independent.
                                       Defaults to True.
            reduction (str, optional): [description]. Defaults to 'sum'.
            n_classes (int, optional): Number of classes. Required only if ground truth is not one hot
        """
        super(LabelSmoothingMultiLabelCrossEntropyLoss, self).__init__()
        self.from_logits=from_logits
        self.label_smoothing=label_smoothing
        self.keep_sum = keep_sum
        self.reduction=reduction
        self.n_classes = n_classes

    def forward(self, y_pred, y_true):
        """
        Calculates the loss.

        Args:
            predictions (List[torch.Tensor]): List with model predictions. Each position is the prediction of a particular task. Shape is (N, n_intents), (N, n_tokens, n_entities)
            targets (List[torch.Tensor]): List with ground truth. Each position is the prediction of a particular task. Shape is (N,), (N, n_tokens)

        Returns:
            torch.Tensor: The loss. If there is any reduction, the shape is (N,). If there is no reduction, shape is (N, n_classes)
        """
        #Single label
        if len(y_true.shape)==1: # Flat it in case it's one-hot encoded
              y_true = torch.nn.functional.one_hot(y_true, num_classes=self.n_classes).float()

        # Label smoothing
        if self.label_smoothing > 0:
            y_true = y_true.clone()
            if self.keep_sum:
              bias = torch.sum(y_true, axis = 1)*self.label_smoothing/(y_true.shape[1]-torch.sum(y_true, axis = 1))
              slope = ((1-self.label_smoothing)-bias)
            else:
              bias = torch.ones(y_true.shape[0])*self.label_smoothing/(y_true.shape[1]-torch.ones(y_true.shape[0]))
              slope = ((1-self.label_smoothing)-bias)
            y_true = torch.multiply(slope.unsqueeze(dim = 1),y_true) + bias.unsqueeze(dim = 1)
        # Check if it comes from logits (no sigmoid applied) or the sigmoid needs to be applied
        # The output of this is Negative LogSigmoid
        if self.from_logits:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true)
        else:
            loss = torch.nn.functional.binary_cross_entropy(y_pred, y_true)

        # Applies reduction method
        if self.reduction=='mean':
            return torch.mean(loss)
        elif self.reduction=='sum':
            return torch.sum(loss)
        else:
            loss = torch.sum(loss, dim = -1)
            return loss