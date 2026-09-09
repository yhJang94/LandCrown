
from collections import namedtuple, Counter
import numpy as np

import torch
import torch.nn.functional as F

import dgl
from dgl.data import (
    load_data, 
    TUDataset, 
    CoraGraphDataset, 
    CiteseerGraphDataset, 
    PubmedGraphDataset
)
from torch.utils.data import Dataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader

from sklearn.preprocessing import StandardScaler
import os
import torch.nn as nn



GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "ogbn-arxiv": DglNodePropPredDataset
}


def preprocess(graph):
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats


def load_dataset(dataset_name):
    assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."
    if dataset_name.startswith("ogbn"):
        dataset = GRAPH_DICT[dataset_name](dataset_name)
    else:
        dataset = GRAPH_DICT[dataset_name]()

    if dataset_name == "ogbn-arxiv":
        graph, labels = dataset[0]
        num_nodes = graph.num_nodes()

        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = preprocess(graph)

        if not torch.is_tensor(train_idx):
            train_idx = torch.as_tensor(train_idx)
            val_idx = torch.as_tensor(val_idx)
            test_idx = torch.as_tensor(test_idx)

        feat = graph.ndata["feat"]
        feat = scale_feats(feat)
        graph.ndata["feat"] = feat

        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
        graph.ndata["label"] = labels.view(-1)
        graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
    else:
        graph = dataset[0]
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
    num_features = graph.ndata["feat"].shape[1]
    num_classes = dataset.num_classes
    return graph, (num_features, num_classes)


class CrownDataset(Dataset):
    def __init__(self, input_path, gt_path, subset='train'):
        
        self.input_path = input_path
        self.gt_path = gt_path
        self.coarse_gt_path = input_path.replace('input', 'gt')  # 실제(마스킹 전) 128점 GT
        self.subset = subset
        
        self.base = os.path.dirname(input_path)
        self.data_list_file = os.path.join(self.base, f'{self.subset}.txt')

        with open(self.data_list_file, 'r') as f:
            file_list = [line.strip() for line in f if line.strip()]

        self.landmark_order = ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'Cusp']

        # target 치아 자체에 landmark annotation이 없는 샘플은 landmark loss의 mask_nodes가
        # 비어 mse가 NaN이 될 위험이 있으므로 미리 제외한다.
        gt_cache = {}
        self.file_list = []
        num_excluded = 0
        for sample in file_list:
            name = sample.split('.')[0]
            random_idx = int(name.split('_')[-1])
            lm_name = name[:-2] if len(str(random_idx)) == 1 else name[:-3]

            if lm_name not in gt_cache:
                gt_cache[lm_name] = IO.get(os.path.join(self.gt_path, lm_name + '.npz'))
            gt_data = gt_cache[lm_name]

            try:
                has_landmark = len(gt_data[random_idx]['class']) > 0
            except Exception:
                has_landmark = False

            if has_landmark:
                self.file_list.append(sample)
            else:
                num_excluded += 1

        if num_excluded > 0:
            print(f"[CrownDataset:{subset}] target 치아에 landmark가 없는 샘플 {num_excluded}개 제외 "
                  f"({len(file_list)} -> {len(self.file_list)})")

    def build_dual_condition_edges(self, tooth_num, landmark_cls):
        # 조건 1: 같은 치아 번호일 때 / 조건 2: 같은 랜드마크 클래스일 때 -> 벡터화(N^2 파이썬 for문 제거)
        same_tooth = tooth_num.unsqueeze(0) == tooth_num.unsqueeze(1)
        same_cls = landmark_cls.unsqueeze(0) == landmark_cls.unsqueeze(1)
        mask = same_tooth | same_cls
        mask.fill_diagonal_(False)  # 자기 자신 제외 (GATv2Conv 등에서 self-loop 설정 가능)

        # mask.nonzero()는 (row, col) 순서로 나오는데, 원래 이중 for문의 (i, j) 순회 순서와 동일함
        idx = mask.nonzero(as_tuple=False)
        edge_src = idx[:, 0]
        edge_dst = idx[:, 1]

        return torch.stack([edge_src, edge_dst], dim=0).long()

    def torch_block_fully_connected(self, input, i, j, block_size=128):
        """
        input: (1536, C) or (1536, 3)
        i, j: block index (0 ~ 11)
        block_size: default 128
        """

        device = input.device

        # 1️⃣ block 시작/끝 index 계산
        start_i = i * block_size
        end_i = (i + 1) * block_size

        start_j = j * block_size
        end_j = (j + 1) * block_size

        # 2️⃣ 해당 block의 global index 생성
        idx_i = torch.arange(start_i, end_i, device=device)
        idx_j = torch.arange(start_j, end_j, device=device)

        # 3️⃣ Fully connected bipartite edge 생성
        # 모든 i 블록 노드 ↔ 모든 j 블록 노드
        src = idx_i.repeat_interleave(block_size)
        dst = idx_j.repeat(block_size)

        # (2, 128*128)
        edge_index = torch.stack([src, dst], dim=0)

        return edge_index

    def normalize(self, input, target, valid_mask=None):

        # 결손/마스킹된 치아의 0 좌표가 섞이면 centroid/scale이 왜곡되므로
        # 유효한(실제 존재하는) 점들만으로 통계량을 계산한다.
        valid_points = input[valid_mask] if valid_mask is not None else input

        # 중심(centroid)을 구하고 각 데이터에서 중심을 뺌
        centroid = np.mean(valid_points, axis=0, keepdims=True)

        input = input - centroid
        new_target = target - centroid

        max_abs = np.max(np.sqrt(np.sum((valid_points - centroid) ** 2, axis=1)))

        # 각 축의 절대 최대값으로 나누어 정규화
        input = input / max_abs
        new_target = new_target / max_abs

        return input, new_target

    def torch_graph(self, input, missing_tooth=None):

        # upper jaw  6,   5,  4,  3,   2,   1,   0,  7,  8,  9, 10, 11, 12, 13
        # lower jaw  27, 26, 25, 24,  23,  22,  21, 14, 15, 16, 17, 18, 19, 20

        i_j = [ [5,6], [5,27], [5,26], [5,25], [5,4],
                [4,5], [4,26], [4,25], [4,24], [4,3],
                [3,4], [3,25], [3,24], [3,23], [3,2],
                [2,3], [2,24], [2,23], [2,22], [2,1],
                [1,2], [1,23], [1,22], [1,21], [1,0],
                [0,1], [0,22], [0,21], [0,14], [0,7],
                [7,1], [7,21], [7,14], [7,15], [7,8],
                [8,7], [8,14], [8,15], [8,16], [8,9],
                [9,8], [9,15], [9,16], [9,17], [9,10],
                [10,9], [10,16], [10,17], [10,18], [10,11],
                [11, 10], [11,17], [11,18], [11,19], [11,12],
                [12, 11], [12,18], [12,19], [12,20], [12,13],
                
                [26,27], [26,6], [26,5], [26,4], [26,25],
                [25,26], [25,5], [25,4], [25,3], [25,24],
                [24,25], [24,4], [24,3], [24,2], [24,23],
                [23,24], [23,3], [23,2], [23,1], [23,22],
                [22,23], [22,2], [22,1], [22,0], [22,21],
                [21,22], [21,1], [21,0], [21,7], [21,14],
                [14,21], [14,0], [14,7], [14,8], [14,15],
                [15,14], [15,7], [15,8], [15,9], [15,16],
                [16,15], [16,8], [16,9], [16,10], [16,17],
                [17,16], [17,9], [17,10], [17,11], [17,18],
                [18,17], [18,10], [18,11], [18,12], [18,19],
                [19,18], [19,11], [19,12], [19,13], [19,20]
                ]

        all_edges = []

        for i, j in i_j:
            # 결손치(missing_tooth)가 낀 치아쌍은 엣지를 만들지 않음 -> 네트워크 연산에서 제외
            if missing_tooth is not None and (missing_tooth[i] or missing_tooth[j]):
                continue

            edge_ij = self.torch_block_fully_connected(input, i, j, 128)
            # edge_ji = self.torch_block_fully_connected(input, j, i, 128)

            all_edges.append(edge_ij)
            # all_edges.append(edge_ji)

        edge_index = torch.cat(all_edges, dim=1)

        # edge_index = torch.stack([src, dst], dim=1)
        return edge_index

    def augmentation(self, input, target, coarse_gt):
        # input(치아 전체), target(landmark), coarse_gt(마스킹 전 128점 GT)에 전부 동일한
        # 회전+스케일을 적용해야 함 - 따로 적용하면 crown 예측 input과 GT가 서로 다른
        # 좌표계에 놓여서 학습이 깨짐.
        theta = np.random.uniform(-np.pi/36, np.pi/36)  # z축 기준 ±5도
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # $z$-axis rotation matrix
        R = np.array([
            [cos_t, -sin_t, 0],
            [sin_t,  cos_t, 0],
            [0,      0,     1]
        ])

        scale = np.random.uniform(0.9, 1.0)

        rotated_input = (input @ R.T) * scale
        rotated_target = (target @ R.T) * scale
        rotated_coarse_gt = (coarse_gt @ R.T) * scale

        return rotated_input, rotated_target, rotated_coarse_gt

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        name = sample.split('.')[0]
        
        random_idx = int(name.split("_")[-1])
        
        if len(str(random_idx)) == 1:
            lm_name = name[:-2]
        else:
            lm_name = name[:-3]
    
        input = IO.get(os.path.join(self.input_path, name + '.npy'))
        gt_data = IO.get(os.path.join(self.gt_path, lm_name + '.npz'))
        coarse_gt = IO.get(os.path.join(self.coarse_gt_path, name + '.npy'))  # 마스킹 전 실제 128점 GT

        # 결손치 마스크: 정규화 전(0값이 그대로 보존된 상태)에서 치아별로 전부 0인지 확인.
        # random_idx(재구성 대상 치아)는 의도적으로 0이므로 결손치가 아님.
        tooth_blocks = input.reshape(28, 128, 3)
        zero_tooth = np.all(tooth_blocks == 0, axis=(1, 2))  # target까지 포함해 0인 치아 전부
        missing_tooth = zero_tooth.copy()
        missing_tooth[random_idx] = False

        # 치아 번호
        tooth_num_list = []
        # 랜드마크 클래스
        landmark_cls_list = []
        # 랜드마크 포인트
        landmark_crd = []
        
        
        for tooth_id in sorted(gt_data.keys()):

            tooth_idx = tooth_id

            # target(random_idx)이 아닌데 crown 자체가 결손(입만 잇몸에 가려진 partial GT 등)이면
            # input에서 이미 제외된 치아이므로 landmark도 동일하게 제외
            if missing_tooth[tooth_idx]:
                continue

            try:
                # 치아 넘버에 대한 coord 좌표
                crd = np.array(gt_data[tooth_id]['coord'])
                cls = gt_data[tooth_id]['class']

                if len(cls) == 0:
                    continue

                for i, cls_type in enumerate(cls):
                    # print(cls_type)
                    idx = self.landmark_order.index(cls_type)

                    tooth_num_list.append(tooth_idx)
                    landmark_cls_list.append(idx)
                    landmark_crd.append(crd[i])


            except:
                continue
            
        # seg_pts = torch.from_numpy(seg_pts)
        # down_seg_pts = torch.from_numpy(down_seg_pts)
        
        input_raw = input.astype(np.float32)
        # 정규화 통계는 target까지 포함해 실제 0-패딩된 점을 모두 제외하고 계산해야 함
        valid_mask = ~np.repeat(zero_tooth, 128)
        input, crd_crd = self.normalize(input_raw, np.array(landmark_crd), valid_mask)
        _, coarse_gt = self.normalize(input_raw, coarse_gt.astype(np.float32), valid_mask)
        if self.subset == 'train':
            # train에만 적용 (validation/test는 결과 재현성을 위해 증강 없이 그대로)
            input, crd_crd, coarse_gt = self.augmentation(input, crd_crd, coarse_gt)

        tooth_num_ts = torch.tensor(tooth_num_list, dtype=torch.long)
        landmark_cls_ts = torch.tensor(landmark_cls_list, dtype=torch.long)
        crd_crd = torch.from_numpy(np.asarray(crd_crd, dtype=np.float32))
        edge = self.build_dual_condition_edges(tooth_num_ts, landmark_cls_ts)
        input = torch.from_numpy(input.astype(np.float32))
        coarse_gt = torch.from_numpy(coarse_gt.astype(np.float32))
        
        landmark_mask_crd = torch.zeros((crd_crd.shape[0], 1), dtype=torch.float64)
        
        mask_indices = (tooth_num_ts == random_idx).nonzero(as_tuple=True)[0]
        
        landmark_mask_crd[mask_indices] = 1.0
        
        # landmark graph
        src = edge[0]
        dst = edge[1]        
        g = dgl.graph((src, dst), num_nodes=crd_crd.shape[0])
        g.ndata["feat"] = crd_crd
        g.ndata["mask"] = landmark_mask_crd
        g.ndata["label"] = landmark_cls_ts
        g.ndata["tooth_num"] = tooth_num_ts
        
        # coarse points graph
        full_edge = self.torch_graph(input, missing_tooth)
        c = dgl.graph((full_edge[0], full_edge[1]), num_nodes=input.shape[0])
        # 결손치 노드는 다른 치아와의 엣지가 전부 제거돼 in-degree 0이 되므로,
        # GATConv의 zero-in-degree 에러를 피하기 위해 self-loop만 추가 (다른 노드로부터는 여전히 정보를 못 받음)
        c = dgl.add_self_loop(c)
        c.ndata["feat"] = input

        return g, c, random_idx, name, coarse_gt

    def __len__(self):
        return len(self.file_list)


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.npz']:
            return cls._read_npz(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)
    
    @classmethod
    def _read_npz(cls, file_path):
        f = np.load(file_path, allow_pickle=True)
        try:
            return f["data"].item()
        except:
            return f

def load_inductive_dataset(dataset_name, batch_size=16):
    if dataset_name == "ppi":
        # data root: override with the DATA_ROOT env var (see README > Data)
        data_root = os.environ.get("DATA_ROOT", "/data/yohan")
        input_path = os.path.join(data_root, "teeth3ds_aligned_input")
        gt_path = os.path.join(data_root, "teeth3ds_land_aligned_npz")

        train_dataset = CrownDataset(input_path=input_path, gt_path=gt_path, subset='train')
        valid_dataset = CrownDataset(input_path=input_path, gt_path=gt_path, subset='test')
        test_dataset = CrownDataset(input_path=input_path, gt_path=gt_path, subset='test')

        train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size)
        valid_dataloader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        eval_train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        # g, name = train_dataset[0]
        num_classes = 28
        # num_features = g.ndata['feat'].shape[1]
        num_features = 256
    else:
        _args = namedtuple("dt", "dataset")
        dt = _args(dataset_name)
        batch_size = 1
        dataset = load_data(dt)
        num_classes = dataset.num_classes

        g = dataset[0]
        num_features = g.ndata["feat"].shape[1]

        train_mask = g.ndata['train_mask']
        feat = g.ndata["feat"]
        feat = scale_feats(feat)
        g.ndata["feat"] = feat

        g = g.remove_self_loop()
        g = g.add_self_loop()

        train_nid = np.nonzero(train_mask.data.numpy())[0].astype(np.int64)
        train_g = dgl.node_subgraph(g, train_nid)
        train_dataloader = [train_g]
        valid_dataloader = [g]
        test_dataloader = valid_dataloader
        eval_train_dataloader = [train_g]
        
    return train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader, num_features, num_classes



def load_graph_classification_dataset(dataset_name, deg4feat=False):
    dataset_name = dataset_name.upper()
    dataset = TUDataset(dataset_name)
    graph, _ = dataset[0]

    if "attr" not in graph.ndata:
        if "node_labels" in graph.ndata and not deg4feat:
            print("Use node label as node features")
            feature_dim = 0
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.ndata["node_labels"].max().item())
            
            feature_dim += 1
            for g, l in dataset:
                node_label = g.ndata["node_labels"].view(-1)
                feat = F.one_hot(node_label, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
        else:
            print("Using degree as node features")
            feature_dim = 0
            degrees = []
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.in_degrees().max().item())
                degrees.extend(g.in_degrees().tolist())
            MAX_DEGREES = 400

            oversize = 0
            for d, n in Counter(degrees).items():
                if d > MAX_DEGREES:
                    oversize += n
            # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
            feature_dim = min(feature_dim, MAX_DEGREES)

            feature_dim += 1
            for g, l in dataset:
                degrees = g.in_degrees()
                degrees[degrees > MAX_DEGREES] = MAX_DEGREES
                
                feat = F.one_hot(degrees, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
    else:
        print("******** Use `attr` as node features ********")
        feature_dim = graph.ndata["attr"].shape[1]

    labels = torch.tensor([x[1] for x in dataset])
    
    num_classes = torch.max(labels).item() + 1
    dataset = [(g.remove_self_loop().add_self_loop(), y) for g, y in dataset]

    print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")

    return dataset, (feature_dim, num_classes)
