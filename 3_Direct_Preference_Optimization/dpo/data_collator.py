import numpy as np
import torch
from transformers import DataCollatorForLanguageModeling
from typing import Any, Dict, List, Optional, Tuple, Union

class DataCollatorForLastTurnOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        instruction_template (`Optional[str]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        last_turn_only (`bool`, *optional*, defaults to `False`): Whether or not to compute the loss only on the last turn response.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_template: Union[str, List[int]],
        instruction_template: Optional[str] = None,
        *args,
        last_turn_only: bool = True,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)
        self.instruction_template = instruction_template
        self.response_template = response_template
        self.last_turn_only = last_turn_only
        self.ignore_index = ignore_index
        
        if type(response_template) == list:
            # The user already provides the token ids
            self.response_token_ids = response_template
        else:
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                    if (
                        self.response_token_ids
                        == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    raise RuntimeError(
                        f'Could not find response key {self.response_token_ids} in token IDs {batch["labels"][i]}'
                    )

                response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                # Make pytorch loss function ignore all tokens up through the end of the response key
                batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []

                for assistant_idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # find the indexes of the start of a response.
                    if (
                        self.response_token_ids
                        == batch["labels"][i][assistant_idx : assistant_idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_idxs.append(assistant_idx + len(self.response_token_ids))

                if len(self.response_token_ids) == 0:
                    raise RuntimeError(
                        f'Could not find response key {self.response_token_ids} in token IDs {batch["labels"][i]}'
                    )

                if type(self.instruction_template) == list:
                    # The user already provides the token ids
                    human_token_ids = self.instruction_template
                else:
                    human_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
            
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                    # find the indexes of the start of a human answer.
                    if human_token_ids == batch["labels"][i][human_idx : human_idx + len(human_token_ids)].tolist():
                        human_token_ids_idxs.append(human_idx)

                if len(human_token_ids_idxs) == 0:
                    raise RuntimeError(
                        f'Could not find response key {human_token_ids} in token IDs {batch["labels"][i]}'
                    )

                if self.last_turn_only:
                    human_token_ids_idxs = [human_token_ids_idxs[-1]]
                    response_token_ids_idxs = [response_token_ids_idxs[-1]]
                    assert len(human_token_ids_idxs) == 1
                    assert len(response_token_ids_idxs) == 1
                for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                    # Make pytorch loss function ignore all non response tokens
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        return batch