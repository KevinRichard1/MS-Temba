import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

from utils import generate_gaussian, video_to_tensor

FEATURE_FPS = 1.87
NUM_CLASSES = 110

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_class_map(train_split, class_map_path):
    """
    Build the 110-class Ego4D Moments vocabulary from primary labels
    """

    if os.path.exists(class_map_path):
        with open(class_map_path, "r") as f:
            class_map = json.load(f)

        if len(class_map) != NUM_CLASSES:
            raise ValueError(
                f"{class_map_path} contains {len(class_map)} classes; "
                f"expected {NUM_CLASSES}"
            )

        return class_map

    data = load_json(train_split)

    class_names = set()

    for video in data["videos"]:
        for clip in video["clips"]:
            for annotation in clip["annotations"]:
                for ann in annotation["labels"]:
                    if ann.get("primary", False):
                        class_names.add(ann["label"])

    class_names = sorted(class_names)

    if len(class_names) != NUM_CLASSES:
        raise ValueError(
            f"Found {len(class_names)} primary classes; "
            f"expected {NUM_CLASSES}"
        )

    class_map = {
        label: idx
        for idx, label in enumerate(class_names)
    }

    with open(class_map_path, "w") as f:
        json.dump(class_map, f, indent=2)

    print(
        f"Created Ego4D class map with "
        f"{len(class_map)} classes: {class_map_path}"
    )

    return class_map


def resample_features(features, target_len):
    """
    features: [T, D]
    returns:  [target_len, D]
    """

    if features.shape[0] == target_len:
        return features

    # [T, D] -> [1, D, T]
    features = features.transpose(0, 1).unsqueeze(0)

    features = F.interpolate(
        features,
        size=target_len,
        mode="linear",
        align_corners=False,
    )

    # [1, D, T] -> [target_len, D]
    return features.squeeze(0).transpose(0, 1)


def make_dataset(
    split_file,
    split,
    root,
    num_classes,
    class_map,
):
    gamma = 0.5
    tau = 4
    ku = 1

    data = load_json(split_file)

    dataset = []

    print('split!!!!', split)

    for video in data["videos"]:

        if video["split"] != split:
            continue

        video_uid = video["video_uid"]

        feature_path = os.path.join(
            root,
            video_uid + ".pt"
        )

        if not os.path.exists(feature_path):
            print(
                f"Skipping {video_uid}: "
                f"missing feature file {feature_path}"
            )
            continue

        # Raw Ego4D feature tensor: [T_raw, 2304]
        v_data = torch.load(
            feature_path,
            map_location="cpu",
        )

        if not torch.is_tensor(v_data):
            v_data = torch.as_tensor(v_data)

        v_data = v_data.float()

        if v_data.ndim != 2:
            raise ValueError(
                f"{feature_path} has shape {v_data.shape}; "
                f"expected [T, D]"
            )

        if v_data.shape[1] != 2304:
            raise ValueError(
                f"{feature_path} has feature dimension "
                f"{v_data.shape[1]}, expected 2304"
            )

        for clip in video["clips"]:
            clip_uid = clip["clip_uid"]

            # Source-video temporal extent represented by this feature file
            video_start_sec = float(
                clip["video_start_sec"]
            )

            video_end_sec = float(
                clip["video_end_sec"]
            )

            source_duration = (
                video_end_sec - video_start_sec
            )

            if source_duration <= 0:
                continue

            # Clip extent within the source-video feature file
            clip_start_sec = float(
                clip["clip_start_sec"]
            )

            clip_end_sec = float(
                clip["clip_end_sec"]
            )

            clip_duration = (
                clip_end_sec - clip_start_sec
            )

            if clip_duration <= 0:
                continue

            # Map clip boundaries onto the raw feature sequence.
            # Raw features are [T_raw, 2304]
            raw_start = int(
                (
                    clip_start_sec
                    / source_duration
                )
                * v_data.shape[0]
            )

            raw_end = int(
                (
                    clip_end_sec
                    / source_duration
                )
                * v_data.shape[0]
            )

            raw_start = max(
                0,
                min(
                    raw_start,
                    v_data.shape[0] - 1,
                ),
            )

            raw_end = max(
                raw_start + 1,
                min(
                    raw_end,
                    v_data.shape[0],
                ),
            )

            win_data = v_data[
                raw_start:raw_end
            ]

            target_len = max(
                int(clip_duration * FEATURE_FPS),
                1,
            )

            win_data = resample_features(
                win_data,
                target_len,
            )

            num_feat = win_data.shape[0]

            # Dense MS-Temba targets
            label = np.zeros(
                (num_feat, num_classes),
                np.float32,
            )

            hmap = np.zeros(
                (num_feat, num_classes),
                np.float32,
            )

            action_lengths = []
            center_loc = []
            num_action = 0

            # Only primary labels belong to the 110-class Moments Query taxonomy
            for annotation in clip["annotations"]:

                for ann in annotation["labels"]:

                    if not ann.get("primary", False):
                        continue

                    label_name = ann["label"]

                    if label_name not in class_map:
                        raise KeyError(
                            f"Unknown primary label: "
                            f"{label_name}"
                        )

                    class_id = class_map[label_name]

                    start_time = float(
                        ann["start_time"]
                    )

                    end_time = float(
                        ann["end_time"]
                    )

                    if end_time <= start_time:
                        continue

                    mid_point = (
                        start_time + end_time
                    ) / 2.0

                    for fr in range(num_feat):

                        timestamp = (
                            fr / FEATURE_FPS
                        )

                        if (
                            timestamp > start_time
                            and timestamp < end_time
                        ):
                            label[fr, class_id] = 1

                        if (
                            (fr + 1) / FEATURE_FPS > mid_point
                            and
                            fr / FEATURE_FPS < mid_point
                        ):

                            center = fr + 1

                            action_duration = int(
                                (
                                    end_time
                                    - start_time
                                )
                                * FEATURE_FPS
                            )

                            radius = int(
                                action_duration / gamma
                            )

                            generate_gaussian(
                                hmap[:, class_id],
                                center,
                                radius,
                                tau,
                                ku,
                            )

                            num_action += 1

                            center_loc.append(
                                [center, class_id]
                            )

                            action_lengths.append(
                                [action_duration]
                            )

            dataset.append(
                {
                    "clip_uid": clip_uid,
                    "video_uid": video_uid,
                    "clip_info": clip,
                    "features": (
                        win_data
                        .numpy()
                        .astype(np.float32)
                    ),
                    "labels": label,
                    "hmap": hmap,
                    "action_lengths": np.asarray(
                        action_lengths
                    ),
                    "num_action": num_action,
                }
            )

    print(
        f"Ego4D {split}: loaded {len(dataset)} clips"
    )

    return dataset


class Ego4D(data_utl.Dataset):
    def __init__(
         self,
        split_file,
        split,
        root,
        batch_size,
        classes,
        num_clips,
        skip,
        class_map,
        is_training=False,
    ):
        self.data = make_dataset(
            split_file,
            split,
            root,
            classes,
            class_map,
        )

        self.split = split
        self.is_training = is_training

        self.split_file = split_file
        self.batch_size = batch_size
        self.root = root

        self.in_mem = {}

        self.num_clips = num_clips
        self.skip = skip

    def __getitem__(self, index):
        entry = self.data[index]

        clip_uid = entry["clip_uid"]

        features = entry["features"]
        labels = entry["labels"]
        hmap = entry["hmap"]
        action_lengths = entry["action_lengths"]
        num_action = entry["num_action"]

        features = features.reshape(
            (
                features.shape[0],
                1,
                1,
                features.shape[-1],
            )
        ).astype(np.float32)

        num_clips = self.num_clips

        if (
            len(features) > num_clips
            and num_clips > 0
        ):

            if self.is_training:
                random_index = random.choice(
                    range(
                        0,
                        len(features) - num_clips
                    )
                )
            else:
                random_index = 0

            features = features[
                random_index:
                random_index + num_clips
            ]

            labels = labels[
                random_index:
                random_index + num_clips
            ]

            hmap = hmap[
                random_index:
                random_index + num_clips
            ]

        return (
            features,
            labels,
            hmap,
            action_lengths,
            [
                clip_uid,
                entry["clip_info"]["clip_end_sec"]
                - entry["clip_info"]["clip_start_sec"],
                num_action,
            ],
        )

    def __len__(self):
        return len(self.data)


class collate_fn_unisize:
    def __init__(self, num_clips):
        self.num_clips = int(num_clips)

    def ego4d_collate_fn_unisize(self, batch):
        max_len = int(self.num_clips)

        new_batch = []

        for b in batch:
            f = np.zeros(
                (
                    max_len,
                    b[0].shape[1],
                    b[0].shape[2],
                    b[0].shape[3],
                ),
                np.float32
            )

            m = np.zeros(
                (max_len,),
                np.float32
            )

            l = np.zeros(
                (
                    max_len,
                    b[1].shape[1]
                ),
                np.float32
            )

            h = np.zeros(
                (
                    max_len,
                    b[2].shape[1]
                ),
                np.float32
            )

            f[:b[0].shape[0]] = b[0]

            m[:b[0].shape[0]] = 1

            l[:b[0].shape[0], :] = b[1]

            h[:b[0].shape[0], :] = b[2]

            new_batch.append(
                [
                    video_to_tensor(f),
                    torch.from_numpy(m),
                    torch.from_numpy(l),
                    b[4],
                    torch.from_numpy(h),
                ]
            )

        return default_collate(new_batch)