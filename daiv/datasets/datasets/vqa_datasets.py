"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch

from daiv.datasets.datasets.base_dataset import BaseDataset


class VQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def collater(self, samples):
        image_list, question_list, answer_list, weight_list = [], [], [], []

        num_answers = []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            #print(sample)
            weight_list.extend(list(sample["weights"].values()))
            #print(weight_list)
            #answers = sample["weights"]
            #print(answers)
            answer_list.extend(sample['weights'])
            #print(answer_list)
            num_answers.append(len(answer_list))

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "answer": answer_list,
            "weight": torch.Tensor(weight_list),
            "n_answers": torch.LongTensor(num_answers),
        }


class VQAEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
