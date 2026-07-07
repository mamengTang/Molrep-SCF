import os
import pickle
import pandas as pd
import scipy.sparse as sps
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from finetune2.features import atom_to_feature_vector, bond_to_feature_vector
import torch
from torch.utils.data import Dataset
import torch_geometric.transforms as T
from finetune2.data import *
from torch_sparse import SparseTensor
from torch_geometric.data import Data
import codecs
from subword_nmt.apply_bpe import BPE
import re
import ast
import string
def parse_atomic_symbols(token):
    """
    从 token 中解析原子符号：
    - 支持普通元素：C, Cl, Br...
    - 支持芳香小写：c, n, o, s, p, b, se, as
    - 支持括号原子：[nH], [13CH3] 等（取元素部分）
    """
    periodic_table = set([  # 定义所有合法的元素符号
        "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr"
    ])

    def norm(sym: str) -> str:
        """
        规范化元素符号：标准化为大写
        """
        sym = sym.strip()
        if sym in ("se", "as"):
            return sym.title()  # Se / As
        if len(sym) == 1:
            return sym.upper()
        return sym[0].upper() + sym[1:].lower()

    out = []

    # # 1) 先抓括号原子：[nH], [13CH3], [O-] ...
    # #    取 isotope 后面的元素符号
    # bracket_syms = re.findall(r"\[(?:\d+)?([A-Za-z]{1,2})", token)
    # out.extend(bracket_syms)

    # 2) 再抓普通元素：C, Cl, Br...以及特殊的小写芳香元素
    # 捕捉单个字符原子或者连续相同字符
    aromatic_syms = []
    for match in re.finditer(r"[A-Z][a-z]?", token):
        symbol = match.group(0)
        start = match.start()
        end = match.end()
        aromatic_syms.append((symbol, start))
    normal_syms2 = []
    for i in aromatic_syms:
        if i[0] not in periodic_table:
            aaaaaa = re.findall(r"[A-Z]?", i[0])
            aaaaaa = [m for m in aaaaaa if m != '']
            assert len(aaaaaa) == 1
            normal_syms2.extend([(aaaaaa[0],i[1])])
        else:
            pass
    out.extend(normal_syms2)

    # 3) 抓芳香的元素：c, n, o, s, p, b, se, as
    aromatic_syms = []
    for match in re.finditer(r"(se|as|[bcnops])", token):
        symbol = match.group(0)
        start = match.start()
        aromatic_syms.append((symbol, start))
    out.extend(aromatic_syms)
    # 4) 归一化 + 过滤合法元素
    out = [item for item in out if item[0] not in (None, '', [])]
    num = [pos for sym, pos in sorted(out, key=lambda x: x[1])]
    num_set = set(num)

    # 这些“非 num 位置”上哪些字符是大写字母
    upper_outside = [(token[i],i) for i in range(len(token))
                    if i not in num_set and token[i] in string.ascii_uppercase]
    out.extend(upper_outside)

    out = [sym for sym, pos in sorted(out, key=lambda x: x[1])]

    out = [norm(x) for x in out]

    # 5) 过滤合法的元素符号
    return [x for x in out if x in periodic_table]

def getface(mol):
    assert isinstance(mol, Chem.Mol)
    bond2id = dict()
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond2id[(i, j)] = len(bond2id)
        bond2id[(j, i)] = len(bond2id)

    num_edge = len(bond2id)
    left = [0] * num_edge
    ssr = Chem.GetSymmSSSR(mol)
    face = [[]]
    for ring in ssr:
        ring = list(ring)

        bond_list = []
        for i, atom in enumerate(ring):
            bond_list.append((ring[i - 1], atom))

        exist = False
        if any([left[bond2id[bond]] != 0 for bond in bond_list]):
            exist = True
        if exist:
            ring = list(reversed(ring))
        face.append(ring)
        for i, atom in enumerate(ring):
            bond = (ring[i - 1], atom)
            if left[bond2id[bond]] != 0:
                bond = (atom, ring[i - 1])
            bondid = bond2id[bond]
            if left[bondid] == 0:
                left[bondid] = len(face) - 1

    return face, left, bond2id

def drug2emb_encoder(smile):
    vocab_path = "./ESPF/drug_codes_chembl_freq_1500.txt"
    sub_csv = pd.read_csv("./ESPF/subword_units_map_chembl_freq_1500.csv")

    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    bpe_codes_drug.close()

    idx2word_d = sub_csv['index'].values
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

    max_d = 50
    t1 = dbpe.process_line(smile).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])

    l = len(i1)
    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))
    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)


def build_atom_to_token_map(order, match_atoms_cnt):
    n_atoms = max(order) + 1
    mapping = [-1] * n_atoms
    p = 0
    for t, c in enumerate(match_atoms_cnt):
        for _ in range(c):
            a = order[p]
            mapping[a] = t
            p += 1
    return mapping

def smiles2graphwithface(mol):

    # mol = Chem.MolFromSmiles(smiles_string)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = np.array(edges_list, dtype=np.int64).T
        edge_attr = np.array(edge_features_list, dtype=np.int64)

        faces, left, _ = getface(mol)
        num_faces = len(faces)
        face_mask = [False] * num_faces
        face_index = [[-1, -1]] * len(edges_list)
        face_mask[0] = True
        for i in range(len(edges_list)):
            inface = left[i ^ 1]
            outface = left[i]
            face_index[i] = [inface, outface]

        nf_node = []
        nf_ring = []
        for i, face in enumerate(faces):
            face = list(set(face))
            nf_node.extend(face)
            nf_ring.extend([i] * len(face))

        face_mask = np.array(face_mask, dtype=bool)
        face_index = np.array(face_index, dtype=np.int64).T
        n_nfs = len(nf_node)
        nf_node = np.array(nf_node, dtype=np.int64).reshape(1, -1)
        nf_ring = np.array(nf_ring, dtype=np.int64).reshape(1, -1)

    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)
        face_mask = np.empty((0), dtype=bool)
        face_index = np.empty((2, 0), dtype=np.int64)
        num_faces = 0
        n_nfs = 0
        nf_node = np.empty((1, 0), dtype=np.int64)
        nf_ring = np.empty((1, 0), dtype=np.int64)

    n_src = list()
    n_tgt = list()
    for atom in mol.GetAtoms():
        n_ids = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(n_ids) > 1:
            n_src.append(atom.GetIdx())
            n_tgt.append(n_ids[:6])
    nums_neigh = len(n_src)
    nei_src_index = np.array(n_src, dtype=np.int64).reshape(1, -1)
    nei_tgt_index = np.zeros((6, nums_neigh), dtype=np.int64)
    nei_tgt_mask = np.ones((6, nums_neigh), dtype=bool)

    for i, n_ids in enumerate(n_tgt):
        nei_tgt_index[: len(n_ids), i] = n_ids
        nei_tgt_mask[: len(n_ids), i] = False

    graph = dict()
    graph["edge_index"] = edge_index
    graph["edge_feat"] = edge_attr
    graph["node_feat"] = x
    graph["num_nodes"] = len(x)

    # we do not use the keyword "face", since "face" is already used by torch_geometric.
    graph["ring_mask"] = face_mask
    graph["ring_index"] = face_index
    graph["num_rings"] = num_faces
    graph["n_edges"] = len(edge_attr)
    graph["n_nodes"] = len(x)

    graph["n_nfs"] = n_nfs
    graph["nf_node"] = nf_node
    graph["nf_ring"] = nf_ring

    graph["nei_src_index"] = nei_src_index
    graph["nei_tgt_index"] = nei_tgt_index
    graph["nei_tgt_mask"] = nei_tgt_mask

    return graph


def espf_tokenize(smile, mol, vocab_path="./ESPF/drug_codes_chembl_freq_1500.txt", subword_map_path="./ESPF/subword_units_map_chembl_freq_1500.csv"):
    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    bpe_codes_drug.close()

    sub_csv = pd.read_csv(subword_map_path)
    idx2word_d = sub_csv['index'].values  # 所有子结构（token）
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

    tokenized_smiles = dbpe.process_line(smile).split()
    match_atoms = []
    match_atoms_cnt = []
    for nn,token in enumerate(tokenized_smiles):
        token_atoms=parse_atomic_symbols(token)
        match_atoms.extend(token_atoms)
        match_atoms_cnt.append(len(token_atoms))
    current_match_atom_cnt = [0]*len(match_atoms_cnt)

    try:
        token_ids = np.asarray([words2idx_d[token] for token in tokenized_smiles])
    except KeyError:
        token_ids = np.array([0])
    
    mol_atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]

    max_length = 50
    seq_length = len(token_ids)

    if seq_length < max_length:
        padded_tokens = np.pad(token_ids, (0, max_length - seq_length), 'constant', constant_values=0)
        attention_mask = [1] * seq_length + [0] * (max_length - seq_length)
    else:
        padded_tokens = token_ids[:max_length]
        attention_mask = [1] * max_length

    return tokenized_smiles, padded_tokens, np.asarray(attention_mask), seq_length


class DGData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if isinstance(value, SparseTensor):
            return (0, 1)
        elif bool(re.search("(index|face)", key)):
            return -1
        elif bool(re.search("(nf_node|nf_ring|nei_tgt_mask)", key)):
            return -1
        return 0

    def __inc__(self, key, value, *args, **kwargs):
        if bool(re.search("(ring_index|nf_ring)", key)):
            return int(self.num_rings.item())
        elif bool(re.search("(index|face|nf_node)", key)):
            return self.num_nodes
        else:
            return 0

class MoleculeDataset(Dataset):
    def __init__(self, data_dir, data_name):

        data = pd.read_csv(os.path.join(data_dir, data_name + '.csv'))
        if 'CHEMBL' in data_name:
            smiles = data['smiles'].to_list()
            labels = data[['y']].values

        elif data_name == 'bbbp':
            smiles = data['smiles'].to_list()
            labels = data[['p_np']]
            labels = labels.replace(0, -1)
            labels = labels.values

        elif data_name == 'clintox':
            smiles = data['smiles'].to_list()
            labels = data[['FDA_APPROVED', 'CT_TOX']]
            labels = labels.replace(0, -1)
            labels = labels.values

        elif data_name == 'muv':
            smiles = data['smiles'].to_list()
            labels = data[['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
                           'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
                           'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']]
            labels = labels.replace(0, -1)
            labels = labels.fillna(0)
            labels = labels.values

        elif data_name == 'sider':
            smiles = data['smiles'].to_list()
            labels = data[['Hepatobiliary disorders',
                           'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
                           'Investigations', 'Musculoskeletal and connective tissue disorders',
                           'Gastrointestinal disorders', 'Social circumstances',
                           'Immune system disorders', 'Reproductive system and breast disorders',
                           'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
                           'General disorders and administration site conditions',
                           'Endocrine disorders', 'Surgical and medical procedures',
                           'Vascular disorders', 'Blood and lymphatic system disorders',
                           'Skin and subcutaneous tissue disorders',
                           'Congenital, familial and genetic disorders',
                           'Infections and infestations',
                           'Respiratory, thoracic and mediastinal disorders',
                           'Psychiatric disorders', 'Renal and urinary disorders',
                           'Pregnancy, puerperium and perinatal conditions',
                           'Ear and labyrinth disorders', 'Cardiac disorders',
                           'Nervous system disorders',
                           'Injury, poisoning and procedural complications']]
            labels = labels.replace(0, -1)
            labels = labels.values

        elif data_name == 'toxcast':
            smiles = data['smiles'].to_list()
            labels = data[list(data.columns)[1:]]
            labels = labels.replace(0, -1)
            labels = labels.fillna(0)
            labels = labels.values

        elif data_name == 'tox21':
            smiles = data['smiles'].to_list()
            labels = data[['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                           'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']]
            labels = labels.replace(0, -1)
            labels = labels.fillna(0)
            labels = labels.values

        elif data_name == 'bace':
            smiles = data['mol'].to_list()
            labels = data[['Class']]
            labels = labels.replace(0, -1)
            labels = labels.values

        elif data_name == 'hiv':
            smiles = data['smiles'].to_list()
            labels = data[['HIV_active']]
            labels = labels.replace(0, -1)
            labels = labels.values

        else:
            raise NotImplementedError

        # convert mol to graph with smiles validity filtering
        self.smiles, self.labels, self.mol_data = [], [], []
        self.transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                # data = eval('mol_to_graph_data_obj_{}'.format(feat_type))(mol)
                self.smiles.append(smi)
                label = labels[i]
                label = np.where(label == -1, 0, label)
                self.labels.append(label)
                # self.mol_data.append(self.transform(data))
        self.num_task = labels.shape[1]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        lb = self.labels[idx]
        data = DGData()
        mol = Chem.MolFromSmiles(smiles)
        graph = smiles2graphwithface(mol)

        assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
        assert len(graph["node_feat"]) == graph["num_nodes"]

        data.__num_nodes__ = int(graph["num_nodes"])

        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))
        x = np.array(atom_features_list, dtype=np.int64)
        # bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = np.array(edges_list, dtype=np.int64).T
        edge_attr = np.array(edge_features_list, dtype=np.int64)
        
        
        data.x = torch.from_numpy(x).to(torch.int64)
        data.edge_index = torch.from_numpy(edge_index).to(torch.int64)
        data.edge_attr = torch.from_numpy(edge_attr).to(torch.int64)
        
        data.smiles_ori = smiles
        data.smiles, data.mask = drug2emb_encoder(smiles)
        data.y = lb

        espf_smiles, data.tokens, data.attention_mask, substructure_num = espf_tokenize(smiles,mol)
            
        
        return data
