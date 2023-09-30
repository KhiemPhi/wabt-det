import pytorch_lightning as pl 
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from loss_fn import CB_loss
from utils import scatter_tSNE
import csv

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, classification_report)






import copy
from typing import Optional, Any, Union, Callable

import torch
from torch import Tensor

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            attn, weight = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            
            x = x + weight
            x = x + self._ff_block(self.norm2(x))
        else:
            attn, weight = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            
            x = self.norm1(x + attn)
            x = self.norm2(x + self._ff_block(x))

        return x, weight

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x, attn_weight = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)
        return self.dropout1(x), attn_weight

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        all_weights = []

        for mod in self.layers:
            output, attn_weight = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            all_weights.append(attn_weight)

        if self.norm is not None:
            output = self.norm(output)

        return output, all_weights

class ContextSentenceTransformer(pl.LightningModule):

    def __init__(self, train_set, test_set, val_set, learning_rate=0.1, batch_size=8, beta=0.99, gamma=2.5, class_num=2, context=True, loss="focal", cross=False, unlabel_set=None):
        super().__init__()         

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
              
               #"facebook/roberta-hate-speech-dynabench-r4-target" 
                #cardiffnlp/twitter-roberta-base-irony
        self.sent_transformer = AutoModel.from_pretrained(model_name)         
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)       
        
        # self.sent_transformer = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", output_attentions=True)         
        # self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2") 
        # 
        dimensions = 768 if "roberta" in model_name else 384           
              
        self.train_set = train_set 
        self.test_set = test_set 
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = 0


        self.val_preds = []
        self.val_labels = []
        self.best_f1 = 0

        self.beta = beta 
        self.gamma = gamma
        self.class_num = class_num
        
        self.ones_prototypes = []

        if loss == "cross-entropy":
            self.cross_entropy = nn.CrossEntropyLoss()
       
        self.similarity_preds = []

        self.context = context 
        self.cross = cross
       
        encoder_layer = TransformerEncoderLayer(d_model=dimensions, nhead=32)
        self.num_layers = 5
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=4)
        #self.transformer_encoder.apply(init_weights)

        if self.cross:
            self.classifier = nn.Linear(dimensions, 2) 
        else: 
            self.classifier = nn.Linear(dimensions*2, 2)           
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        self.loss = loss
        self.margin = 5
    
    
    def train_dataloader(self):
        """
            Returns the train data-loader, MANDATORY for PyTorch-Lightning to work.
                (1) Set num_workers equal to number of CPUs
                (2) Shuffle= True as this is the train_set
                (3) collate_fn: To gather all the batches
                (4) pin_memory: True to increase performance
        """
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
    
    def test_dataloader(self):
        """
            Returns the test data-loader, MANDATORY for PyTorch-Lightning to work.
                (1) Set num_workers equal to number of CPUs
                (2) Shuffle= False as this is the test_set
                (3) collate_fn: To gather all the batches
                (4) pin_memory: True to increase performance
        """
        
        test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
      
        return test_loader
    
    def val_dataloader(self):
        """
            Returns the val data-loader, MANDATORY for PyTorch-Lightning to work.
                (1) Set num_workers equal to number of CPUs
                (2) Shuffle= True as this is the val_set
                (3) collate_fn: To gather all the batches
                (4) pin_memory: True to increase performance
        """
        
        test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
       
        return test_loader
    
    def configure_optimizers(self):   
        """
            Configure the optimizers and LR scheduler to better train the model
                (1) Optimizer: Can be Adam or SGD (etc)
                (2) Scheduler: Step-wise LR
        """     
        params =  self.parameters()
        opt =  torch.optim.Adam(params, lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.1)
        
        return [opt], [sch]
    
    def get_comment_tokens(self, sentences, device):
        with torch.no_grad():         
            # ToDO: Smart batching   
            inputs = self.tokenizer(list(sentences), return_tensors="pt", padding='max_length', max_length=256, truncation=True)       
            for key in inputs.keys():    
                inputs[key] = inputs[key].to(device)
           
        
        return inputs   
    
    def max_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return torch.max(token_embeddings, 1)[0]
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embs(self, comments, labels):
        comment_tokens = self.get_comment_tokens(comments, labels.device)
        comment_embs = self.sent_transformer(**comment_tokens)
       
        sentence_embeddings = self.max_pooling(comment_embs, comment_tokens['attention_mask'])
        
        return sentence_embeddings
    
    
    def inference(self, whatabout, train=True):
        
        comments, labels, context_comment, context_labels, topic = whatabout
       
        final_sim_scores = []
        final_gt_scores = []
        context_embs = []

        
        if self.cross:
           
            cross_enc = list(zip(comments, context_comment))
            context_tokens =  self.get_comment_tokens(cross_enc, labels.device)
            context_embs = self.sent_transformer(**context_tokens)["pooler_output"]
        else: 

            # To-do: run until find best

            # 1. Push Comment
            comment_embs = self.get_embs(comments, labels)
            context_comment = np.vstack(context_comment)   
            

            # Save the triplet into csv-file? 
            


            context_labels = torch.vstack(context_labels)
            final_label = [] 
            sim_loss = []
            #neg_context = self.get_embs(context_comment[0, :], labels)
            #pos_context = self.get_embs(context_comment[1, :], labels)           

            context_single_embs = self.get_embs(context_comment.flatten(), labels)
            
            all_comments_for_attention = torch.vstack((comment_embs, context_single_embs))
            attention_comments, attn_weight = self.transformer_encoder(all_comments_for_attention) 

            attn_weight[-1] = attn_weight[-1].fill_diagonal_(0)

            # --- use cosine similiarty 
            #sim_weights = F.cosine_similarity(all_comments_for_attention, all_comments_for_attention.unsqueeze(1), dim=-1)
            #sim_weights = sim_weights.fill_diagonal_(0)
            
            
            
            arg_max_weights = torch.argmax(attn_weight[-1], dim=1)[0:comment_embs.shape[0]]
            #arg_max_weights = (torch.rand_like(arg_max_weights, dtype=float)*attention_comments.shape[0]-1).round().long()
            #arg_max_weights = torch.argmax(sim_weights,dim=1)[0:comment_embs.shape[0]]
            
            context_embs = all_comments_for_attention[arg_max_weights] * attention_comments[arg_max_weights]
            comment_embs = comment_embs * attention_comments[0:comment_embs.shape[0]]
              
            
            classifier_embs = torch.hstack((comment_embs, context_embs))
           
            whataboutism_logits = self.classifier(classifier_embs)
            whataboutism_labels = torch.argmax(whataboutism_logits, dim=1).cpu().tolist()
            
           
      

        if train:            
            return whataboutism_logits
        else: 
            return whataboutism_logits, classifier_embs, final_label, whataboutism_labels

    def calculate_loss(self, whataboutism_logits, labels, labels_occurence):
        if self.loss == "softmax" or self.loss == "focal":
            loss = CB_loss(labels, whataboutism_logits, labels_occurence, self.class_num, loss_type=self.loss, beta=self.beta, gamma=self.gamma, device=labels.device)            
        else: 
            loss = self.cross_entropy(whataboutism_logits, labels)
        return loss
    
    def on_train_epoch_start(self):
        self.avg_sim_scores = []
        self.ones_prototypes = []
        self.zeros_prototypes = []
        self.avg_distance_positive = []
        self.avg_distance_negative = []


    def training_step(self, batch: dict, _batch_idx: int):        
        comments, labels, opp_comment, context_labels, topic = batch     
        # one comment can have one of five contexts        
        samples_per_cls = list(np.bincount(labels.cpu().numpy().astype('int64')))  
        whataboutism_logits = self.inference(batch)  
        classifier_loss = self.calculate_loss(whataboutism_logits, labels, samples_per_cls)
        #aux_loss = self.calculate_loss(aux_logits, labels, labels_occurence )
       
        return  classifier_loss 
    
    def on_train_epoch_end(self):        
        self.epochs += 1  
       
    

    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_labels = []
        self.similarity_preds = []
        self.val_embs = []
        self.val_comments = []
        self.val_probs = []

        self.train_preds = []
        self.train_labels = []
    
    def on_test_epoch_start(self):
        self.val_preds = []
        self.val_labels = []
        self.similarity_preds = []
        self.val_embs = []
        self.val_comments = []
        self.val_probs = []

        self.train_preds = []
        self.train_labels = []
      
    def test_step(self, batch: dict, _batch_idx: int):
        comments_test, labels_test, opp_comment_test, context_labels_test, topic = batch
      
        whataboutism_logits_test, context_embs, final_sim_labels_test, pred_labels = self.inference(batch, train=False)  
        #whataboutism_logits_train, aux_logits, context_embs_train, final_sim_labels_train = self.inference(batch["test"], train=False)  
        
        probs = torch.softmax(whataboutism_logits_test, dim=1)[:, 1].cpu().tolist()
        
        self.val_preds.extend(pred_labels)
        self.val_labels.extend(labels_test.cpu().tolist())        
        self.val_embs.extend(context_embs.cpu().tolist())
        self.val_comments.extend(comments_test)
        self.val_probs.extend(probs)

        self.similarity_preds.extend(final_sim_labels_test)

        samples_per_cls = list(np.bincount(labels_test.cpu().numpy().astype('int64')))  
        classifier_loss = self.calculate_loss(whataboutism_logits_test, labels_test, samples_per_cls)
        self.log("test-loss", classifier_loss, prog_bar=True, on_step=True, on_epoch=False, logger=True, sync_dist=True)
    
    def on_test_epoch_end(self):
        self.val_accuracy = accuracy_score(self.val_labels, self.val_preds)*100
        self.val_f1 = f1_score(self.val_labels, self.val_preds)*100

        #self.train_f1 = f1_score(self.train_labels, self.train_preds)*100

        #self.sim_f1 = f1_score(self.val_labels, self.similarity_preds)*100
        
        
        if self.val_f1 > self.best_f1: 

            self.csv_record = open('vis/validation_results_1342.csv', 'w')
            self.writer = csv.writer(self.csv_record)

            self.report = classification_report(self.val_labels, self.val_preds, target_names=["Non-Whataboutism", "Whataboutism"])
          
            with open('{}.txt'.format('vis/validation_acc_tab'), 'w') as f:
                print(self.report, file=f)
        
            self.best_f1 = self.val_f1
            self.val_embs = np.array(self.val_embs)

            # Visualise the results when best is beaten
            path = "vis/tSNE/test-tSNE-epoch-" + str(self.epochs) + ".jpg"
        
            scatter_tSNE(self.val_embs, np.array(self.val_labels), file_path= path )
            

            # Visualize the wrong results
            wrong_comments = []
            for test_comment, test_label, test_pred, test_prob in zip(self.val_comments, self.val_labels, self.val_preds, self.val_probs):                        
                if test_label == 1 and test_pred == 0:
                    self.writer.writerow([test_comment, test_label, test_pred, test_prob,  "False Neg"])
                    wrong_comments.append(test_comment)
                elif test_label == 0 and test_pred == 1:
                    self.writer.writerow([test_comment, test_label, test_pred, test_prob,  "False Pos"])
                    wrong_comments.append(test_comment)
            self.log("best-epoch", self.epochs, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)



        self.log("validation-acc", torch.tensor([self.val_accuracy]), prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("validation-f1", self.val_f1, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        #self.log("sim-f1", self.sim_f1, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("best-f1", self.best_f1, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
    
    def validation_step(self,  batch: dict, _batch_idx: int):       
       
        comments_test, labels_test, opp_comment_test, context_labels_test, topic = batch
      
        whataboutism_logits_test, context_embs, final_sim_labels_test, pred_labels = self.inference(batch, train=False)  
        #whataboutism_logits_train, aux_logits, context_embs_train, final_sim_labels_train = self.inference(batch["test"], train=False)  
        
        probs = torch.softmax(whataboutism_logits_test, dim=1)[:, 1].cpu().tolist()
        
        self.val_preds.extend(pred_labels)
        self.val_labels.extend(labels_test.cpu().tolist())        
        self.val_embs.extend(context_embs.cpu().tolist())
        self.val_comments.extend(comments_test)
        self.val_probs.extend(probs)

        self.similarity_preds.extend(final_sim_labels_test)

        samples_per_cls = list(np.bincount(labels_test.cpu().numpy().astype('int64')))  
        classifier_loss = self.calculate_loss(whataboutism_logits_test, labels_test, samples_per_cls)
        self.log("val-loss", classifier_loss, prog_bar=True, on_step=True, on_epoch=False, logger=True, sync_dist=True)

        
    def on_validation_epoch_end(self):
        self.val_accuracy = accuracy_score(self.val_labels, self.val_preds)*100
        self.val_f1 = f1_score(self.val_labels, self.val_preds)*100

        #self.train_f1 = f1_score(self.train_labels, self.train_preds)*100

        #self.sim_f1 = f1_score(self.val_labels, self.similarity_preds)*100
        
        if self.val_f1 > self.best_f1: 

            self.csv_record = open('vis/validation_results_1342.csv', 'w')
            self.writer = csv.writer(self.csv_record)

            report = classification_report(self.val_labels, self.val_preds, target_names=["Non-Whataboutism", "Whataboutism"])
          
            with open('{}.txt'.format('vis/validation_acc_tab'), 'w') as f:
                print(report, file=f)
        
            self.best_f1 = self.val_f1
            self.val_embs = np.array(self.val_embs)

            # Visualise the results when best is beaten
            path = "vis/tSNE/test-tSNE-epoch-" + str(self.epochs) + ".jpg"
            scatter_tSNE(self.val_embs, np.array(self.val_labels), file_path= path )
            
            # Visualize the wrong results
            wrong_comments = []
            for test_comment, test_label, test_pred, test_prob in zip(self.val_comments, self.val_labels, self.val_preds, self.val_probs):                        
                if test_label == 1 and test_pred == 0:
                    self.writer.writerow([test_comment, test_label, test_pred, test_prob,  "False Neg"])
                    wrong_comments.append(test_comment)
                elif test_label == 0 and test_pred == 1:
                    self.writer.writerow([test_comment, test_label, test_pred, test_prob,  "False Pos"])
                    wrong_comments.append(test_comment)
           

        self.log("validation-acc", torch.tensor([self.val_accuracy]), prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("validation-f1", self.val_f1, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        #self.log("sim-f1", self.sim_f1, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("best-f1", self.best_f1, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
       
        
class SentenceTransformer(pl.LightningModule):
    
    def __init__(self, train_set, test_set, val_set, learning_rate=0.1, batch_size=8, beta=0.99, gamma=2.5, class_num=2, context=True, loss="focal"):
        super().__init__()         
       
        model_name = "bert-base-uncased"
        self.sent_transformer = AutoModel.from_pretrained(model_name)         
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)             #cardiffnlp/twitter-roberta-base-irony
              
        self.train_set = train_set 
        self.test_set = test_set 
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = 0

        self.val_preds = []
        self.val_labels = []
        self.best_f1 = 0

        self.beta = beta 
        self.gamma = gamma
        self.class_num = class_num
        
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.similarity_preds = []

        self.context = context 

       
        self.classifier = nn.Linear(768, 2) #MLP Classifier
        self.cross_entropy = nn.CrossEntropyLoss()
      

        self.loss = loss
    
    def train_dataloader(self):
        """
            Returns the train data-loader, MANDATORY for PyTorch-Lightning to work.
                (1) Set num_workers equal to number of CPUs
                (2) Shuffle= True as this is the train_set
                (3) collate_fn: To gather all the batches
                (4) pin_memory: True to increase performance
        """
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
    
    def test_dataloader(self):
        """
            Returns the test data-loader, MANDATORY for PyTorch-Lightning to work.
                (1) Set num_workers equal to number of CPUs
                (2) Shuffle= False as this is the test_set
                (3) collate_fn: To gather all the batches
                (4) pin_memory: True to increase performance
        """
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
    
    def val_dataloader(self):
        """
            Returns the val data-loader, MANDATORY for PyTorch-Lightning to work.
                (1) Set num_workers equal to number of CPUs
                (2) Shuffle= True as this is the val_set
                (3) collate_fn: To gather all the batches
                (4) pin_memory: True to increase performance
        """
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
    
    def configure_optimizers(self):   
        """
            Configure the optimizers and LR scheduler to better train the model
                (1) Optimizer: Can be Adam or SGD (etc)
                (2) Scheduler: Step-wise LR
        """     
        params =  self.parameters()
        opt =  torch.optim.Adam(params, lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.1)
        
        return [opt], [sch]
    
    def get_comment_tokens(self, sentences, device):
        with torch.no_grad():         
            # ToDO: Smart batching   
            inputs = self.tokenizer(list(sentences), return_tensors="pt", padding='max_length', max_length=256, truncation=True)       
            for key in inputs.keys():    
                inputs[key] = inputs[key].to(device)
        
        return inputs
    
    def max_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return torch.max(token_embeddings, 1)[0]
    

    def training_step(self, batch: dict, _batch_idx: int):
       
        comments, labels = batch
     
        comment_tokens = self.get_comment_tokens(comments, labels.device)
       
        comment_embs = self.sent_transformer(**comment_tokens)
        
        comment_embs = self.max_pooling(comment_embs["last_hidden_state"], comment_tokens['attention_mask'])
        
        
        whataboutism_logits = self.classifier(comment_embs) 
    
        
        labels_occurence = list(np.bincount(labels.cpu().numpy()))
        if self.loss == "softmax" or self.loss == "focal":
            loss = CB_loss(labels, whataboutism_logits, labels_occurence, self.class_num, loss_type=self.loss, beta=self.beta, gamma=self.gamma, device=labels.device)
        else: 
            loss = self.cross_entropy(whataboutism_logits, labels)

        return loss
    
    def on_train_epoch_end(self):
        self.epochs += 1    

    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_labels = []
        self.similarity_preds = []
        self.val_embs = []
        self.val_comments = []
        self.val_probs = []
    
    def validation_step(self,  batch: dict, _batch_idx: int):
        comments, labels = batch
        comment_tokens = self.get_comment_tokens(comments, labels.device)
        comment_embs = self.sent_transformer(**comment_tokens)
        comment_embs = comment_embs["last_hidden_state"]
        comment_embs = self.max_pooling(comment_embs, comment_tokens['attention_mask'])
       
       
        logits = self.classifier(comment_embs) 
          
        preds = torch.argmax(logits, dim=1).flatten()
        probs = torch.softmax(logits, dim=1).flatten()
        self.val_preds.extend(preds.cpu().tolist())
        self.val_labels.extend(labels.cpu().tolist())        

        self.val_comments.extend(comments)
        self.val_probs.extend(probs.cpu().tolist())
        self.val_embs.extend(comment_embs.cpu().tolist())
    
    def on_validation_epoch_end(self):
        self.val_accuracy = accuracy_score(self.val_labels, self.val_preds)*100
        self.val_f1 = f1_score(self.val_labels, self.val_preds)*100
        
        if self.val_f1 >= self.best_f1: 

            self.csv_record = open('vis/validation_results_1615.csv', 'w')
            self.writer = csv.writer(self.csv_record)
            self.best_epoch = self.epochs

            report = classification_report(self.val_labels, self.val_preds, target_names=["Non-Whataboutism", "Whataboutism"])
          
            with open('{}.txt'.format('vis/validation_acc_tab'), 'w') as f:
                print(report, file=f)
        
            self.best_f1 = self.val_f1
            self.val_embs = np.array(self.val_embs)

            # Visualise the results when best is beaten
            # path = "vis/tSNE/test-tSNE-epoch-" + str(self.epochs) + ".jpg"
            # scatter_tSNE(self.val_embs, np.array(self.val_labels), file_path= path )

            # Visualize the wrong results
            wrong_comments = []
            self.writer.writerow(["Comment", "Label", "Predicted (0=Non-Wabt, 1=Wabt)", "Whataboutism Probability", "Error Type"])
            for test_comment, test_label, test_pred, test_prob in zip(self.val_comments, self.val_labels, self.val_preds, self.val_probs):                        
                if test_label == 1 and test_pred == 0:
                    self.writer.writerow([test_comment, test_label, test_pred, test_prob,  "False Neg"])
                    wrong_comments.append(test_comment)
                elif test_label == 0 and test_pred == 1:
                    self.writer.writerow([test_comment, test_label, test_pred, test_prob,  "False Pos"])
                    wrong_comments.append(test_comment)



        self.log("validation-acc", torch.tensor([self.val_accuracy]), prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("validation-f1", self.val_f1, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("best-f1", self.best_f1, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("best-epoch", self.best_epoch, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
    

