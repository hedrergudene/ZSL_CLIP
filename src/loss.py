# Requirements
import torch

# CrossEntropyLoss
class CELoss(torch.nn.Module):
    """
    Implementation of CrossEntropyLoss minimisation function with class weighting and
    multilabel features.
    """
    def __init__(self,
                 temperature:float=1.,
                 from_logits:bool = True,
                 multilabel:bool=False,
                 reduction:str = 'mean',
                 n_classes:int = None,
                 class_weights:torch.Tensor=None,
                 device:str='cuda',
                 )->None:
        """
        Args:
        """
        super(CELoss, self).__init__()
        # Validations
        if not torch.is_tensor(class_weights) and (class_weights is not None):
            raise TypeError("Class weights type is not a torch.Tensor. Got {}"
                            .format(type(class_weights)))
        if class_weights is not None:
            if len(class_weights.shape)!=1:
                raise TypeError("Class weights do not have the right shape. Got shape {}"
                                .format(len(class_weights.shape)))
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError("Reduction should be one of these values: {}"
                            .format(', '.join(['mean', 'sum', 'none'])))
        # Loss config settings
        self.from_logits=from_logits
        self.multilabel = multilabel
        self.reduction=reduction
        # Loss parameters
        self.temperature = temperature
        self.class_weights = torch.ones((n_classes)).unsqueeze(dim=-1).to(device) if class_weights is None else class_weights.unsqueeze(dim=-1).to(device)
        self.n_classes = n_classes
        self.eps = 1e-6

    def forward(self,
                input:torch.Tensor,
                target:torch.Tensor,
                )->torch.Tensor:
        """
        Calculates the loss.

        Args:
            input (torch.Tensor): Batch of model predictions. The last dimension must contain the proability distribution for each
                                  of the classes; i.e., input shape=(batch_size, n_classes) for IntentClassification problems, and
                                  input shape=(batch_size, max_postion_embeddings, n_classes) for NER problems.
            target (torch.Tensor): Batch containing ground truth, either in shape of binarised or one-hot encoded labels.

        Returns:
            torch.Tensor: Loss tensor. If there is any reduction, output is 0-dimensional. If there is no reduction, loss is provided
                          element-wise through the batch.
        """
        # Part I: Validations
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))

        # Part II: Labels preprocessing
        # One-hot encode labels
        if len(target.shape) < len(input.shape): target = torch.nn.functional.one_hot(target, num_classes=self.n_classes).float()

        # Part III: Compute loss
        loss = self.compute_loss(input, target)

        # Part IV:  Apply reduction method
        if self.reduction=='mean':
            return torch.mean(loss)
        elif self.reduction=='sum':
            return torch.sum(loss)
        else: # Sum is already done in class weighting
            #loss = torch.sum(loss, dim = -1)
            return loss

    def compute_loss(self,
                     input:torch.Tensor,
                     target:torch.Tensor,
                     )->torch.Tensor:
        if self.from_logits:
            input_norm = torch.nn.functional.logsigmoid(input) if self.multilabel else torch.nn.functional.log_softmax(input/self.temperature, dim=-1)
        else:
            input_norm = torch.log(input)
        # Compute the actual focal loss and weights classes
        focal_weights = - target * input_norm
        if len(focal_weights.shape)<3:
            focal_weights = torch.unsqueeze(focal_weights, dim=1)
        focal_loss = torch.bmm(focal_weights, self.class_weights.repeat(input.shape[0],1,1))
        return torch.squeeze(focal_loss)


# CLIPLoss
class CLIPLoss(torch.nn.Module):
    """
    Implementation of CLIP model minimisation function
    """
    def __init__(self,
                temperature:float=.05,
                )->None:
        """
        Args:
        """
        super(CLIPLoss, self).__init__()
        # Validations
        self.temperature = temperature
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self,
                input:torch.Tensor,
                )->torch.Tensor:
        """
        Calculates the loss.

        Args:
            input (torch.Tensor): Batch of model predictions. It consists of a dictionary with, respectively, text and vision embeddings.

        Returns:
            torch.Tensor: Loss tensor. If there is any reduction, output is 0-dimensional. If there is no reduction, loss is provided
                          element-wise through the batch.
        """
        # logits[i][j] is the dot_similarity(caption_i, image_j)
        logits = (input['text_emb'] @ input['vision_emb'].T)/self.temperature
        # text_similarity[i][j] is the dot_similarity(caption_i, caption_j).
        text_similarity = input['text_emb'] @input['text_emb'].T
        # images_similarity[i][j] is the dot_similarity(image_i, image_j).
        images_similarity = input['vision_emb'] @input['vision_emb'].T
        # targets[i][j] = avarage dot_similarity(caption_i, caption_j) and dot_similarity(image_i, image_j).
        targets = torch.nn.functional.softmax(
            (text_similarity + images_similarity) / (2 * self.temperature),
            dim=-1,
        )
        # Compute the loss for the captions using crossentropy
        text_loss = self.loss_fn(
            input=logits, target=targets,
        )
        # Compute the loss for the images using crossentropy
        vision_loss = self.loss_fn(
            input=logits.T, target=targets.T,
        )
        # Return the mean of the loss over the batch
        return {'text_loss':text_loss, 'vision_loss':vision_loss, 'summary':.5*(text_loss+vision_loss)}