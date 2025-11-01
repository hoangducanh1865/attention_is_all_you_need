import torch
import torch.nn as nn
from functools import partial # @STAR
class Utils:
    @staticmethod
    def _init_weights_(module):
        """
        Simple weight initialization taken directly from the HuggingFace
        `modeling_roberta.py` implementation!
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
def translation_collate_fn(src_tokenizer,tgt_tokenizer,batch):
        src_ids=[torch.tensor(i['src_ids']) for i in batch]
        tgt_ids=[torch.tensor(i['tgt_ids']) for i in batch]
        src_pad_token=src_tokenizer.pad_token_id # pad_token_id is an attribute of AutoTokenizer
        src_padded_ids=torch.nn.utils.rnn.pad_sequence(src_ids,batch_first=True,padding_value=src_pad_token)
        src_pad_mask=(src_padded_ids!=src_pad_token)
        tgt_pad_token=tgt_tokenizer.special_tokens['[PAD]']
        tgt_padded_ids=torch.nn.utils.rnn.pad_sequence(tgt_ids,batch_first=True,padding_value=tgt_pad_token)
        
        # @QUESTION: why do we need to use clone() here?
        # @ANSWER: We need to use clone() here since PyTorch creates a view that shares the same memory as the original tensor -> both new tensors point to the same data in memory -> during backpropagation, gradients might accumulate incorrectly; and if we modify one tensor, if will affect the other
        tgt_input_padded_ids=tgt_padded_ids[:,:-1].clone() # First dimension is batch_size, in second dimension
        tgt_output_padded_ids=tgt_padded_ids[:,1:].clone()
        tgt_output_padded_ids[tgt_output_padded_ids==tgt_pad_token]=-100 # Assign -100 for padding token so that when calculating the CrossEntropyLoss, any sample which has label (use the word "label" here since this tgt_output_padded_ids should be the output of the model) -100 will be automatically ignored
        input_tgt_pad_mask=(tgt_input_padded_ids!=tgt_pad_token)
        '''tgt_output_padded_ids=(tgt_output_padded_ids!=tgt_pad_token)''' # We do not need this padding mask
        batch={
            'src_input_ids':src_padded_ids,
            'src_pad_mask':src_pad_mask,
            'tgt_input_ids':tgt_input_padded_ids,
            'tgt_pad_mask':input_tgt_pad_mask,
            'tgt_output_ids':tgt_output_padded_ids
        }
        return batch
def translation_collator(src_tokenizer,tgt_tokenizer):
    return partial(translation_collate_fn,src_tokenizer,tgt_tokenizer) # @QUESTION