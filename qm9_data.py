import os
import os.path as osp
import sys
from typing import Callable, List, Optional
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
import re
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
import ast
import string
import codecs
from subword_nmt.apply_bpe import BPE
import pandas as pd
from torch_geometric.io import fs
from torch_geometric.utils import one_hot, scatter
from ogb.utils.features import (atom_to_feature_vector, bond_to_feature_vector)
HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}

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
    atom_count = len(mol_atoms)
    atom_substructure_mapping = [-1] * atom_count
    order = ast.literal_eval(mol.GetProp('_smilesAtomOutputOrder'))
    if len(match_atoms) != len(mol_atoms) or  len(mol_atoms) != len(order) or len(match_atoms) != len(order):
        return None, None, None, None, None

    atom_substructure_mapping = build_atom_to_token_map(order, match_atoms_cnt)

    # atom_idx = 0
    # temp_token_pos = 0

    # for atom_idx in range(0, atom_count):
    #     flag = False
    #     current_token_pos = temp_token_pos
    #     for token in tokenized_smiles[current_token_pos:]:

    #         if current_match_atom_cnt[temp_token_pos] >= match_atoms_cnt[temp_token_pos]:
    #             temp_token_pos += 1
    #         else:
    #             token_atoms = match_atoms[temp_token_pos]
    #             if mol_atoms[atom_idx] in token_atoms:
    #                 atom_substructure_mapping[atom_idx] = temp_token_pos
    #                 current_match_atom_cnt[temp_token_pos] += 1
    #                 match_atoms[temp_token_pos].remove(mol_atoms[atom_idx])
    #                 flag = True
    #                 break
    #             temp_token_pos += 1
    #     if not flag:
    #         temp_token_pos = current_token_pos

    max_length = 50
    seq_length = len(token_ids)

    if seq_length < max_length:
        padded_tokens = np.pad(token_ids, (0, max_length - seq_length), 'constant', constant_values=0)
        attention_mask = [1] * seq_length + [0] * (max_length - seq_length)
    else:
        padded_tokens = token_ids[:max_length]
        attention_mask = [1] * max_length

    return tokenized_smiles, padded_tokens, np.asarray(attention_mask), seq_length, atom_substructure_mapping
class QM9_our(InMemoryDataset):
    r"""The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
    Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
    about 130,000 molecules with 19 regression targets.
    Each molecule includes complete spatial information for the single low
    energy conformation of the atoms in the molecule.
    In addition, we provide the atom features from the `"Neural Message
    Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.

    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | Target | Property                         | Description                                                                       | Unit                                        |
    +========+==================================+===================================================================================+=============================================+
    | 0      | :math:`\mu`                      | Dipole moment                                                                     | :math:`\textrm{D}`                          |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 1      | :math:`\alpha`                   | Isotropic polarizability                                                          | :math:`{a_0}^3`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 2      | :math:`\epsilon_{\textrm{HOMO}}` | Highest occupied molecular orbital energy                                         | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 3      | :math:`\epsilon_{\textrm{LUMO}}` | Lowest unoccupied molecular orbital energy                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 4      | :math:`\Delta \epsilon`          | Gap between :math:`\epsilon_{\textrm{HOMO}}` and :math:`\epsilon_{\textrm{LUMO}}` | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 5      | :math:`\langle R^2 \rangle`      | Electronic spatial extent                                                         | :math:`{a_0}^2`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 6      | :math:`\textrm{ZPVE}`            | Zero point vibrational energy                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 7      | :math:`U_0`                      | Internal energy at 0K                                                             | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 8      | :math:`U`                        | Internal energy at 298.15K                                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 9      | :math:`H`                        | Enthalpy at 298.15K                                                               | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 10     | :math:`G`                        | Free energy at 298.15K                                                            | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 11     | :math:`c_{\textrm{v}}`           | Heat capavity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 12     | :math:`U_0^{\textrm{ATOM}}`      | Atomization energy at 0K                                                          | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 13     | :math:`U^{\textrm{ATOM}}`        | Atomization energy at 298.15K                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 14     | :math:`H^{\textrm{ATOM}}`        | Atomization enthalpy at 298.15K                                                   | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 15     | :math:`G^{\textrm{ATOM}}`        | Atomization free energy at 298.15K                                                | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 16     | :math:`A`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 17     | :math:`B`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 18     | :math:`C`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+

    .. note::

        We also provide a pre-processed version of the dataset in case
        :class:`rdkit` is not installed. The pre-processed version matches with
        the manually processed version as outlined in :meth:`process`.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - #graphs
          - #nodes
          - #edges
          - #features
          - #tasks
        * - 130,831
          - ~18.0
          - ~37.3
          - 11
          - 19
    """  # noqa: E501

    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    def atomref(self, target: int) -> Optional[Tensor]:
        if target in atomrefs:
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[target])
            return out.view(-1, 1)
        return None

    @property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit  # noqa
            return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']
        except ImportError:
            return ['qm9_v3.pt']

    @property
    def processed_file_names(self) -> str:
        return 'geometric_data_processed.pt'

    def download(self) -> None:
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(osp.join(self.raw_dir, '3195404'),
                      osp.join(self.raw_dir, 'uncharacterized.txt'))
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self) -> None:
        try:
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')  # type: ignore
            WITH_RDKIT = True

        except ImportError:
            WITH_RDKIT = False

        if not WITH_RDKIT:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)

            data_list = fs.torch_load(self.raw_paths[0])
            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.save(data_list, self.processed_paths[0])
            return

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        with open(self.raw_paths[1]) as f:
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in f.read().split('\n')[1:-1]]
            y = torch.tensor(target, dtype=torch.float)
            y = torch.cat([y[:, 3:], y[:, :3]], dim=-1)
            y = y * conversion.view(1, -1)

        with open(self.raw_paths[2]) as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)

        data_list = []
        for i, mol_from_sdf in enumerate(tqdm(suppl)):
            data = Data()
            if i in skip:
                continue
            try:
                mol_from_sdf = Chem.RemoveHs(mol_from_sdf) 
            except:
                continue
            

            conf = mol_from_sdf.GetConformer()
            pos = conf.GetPositions()
            pos_from_sdf = torch.tensor(pos, dtype=torch.float)
            smiles = Chem.MolToSmiles(mol_from_sdf, isomericSmiles=True)
            mol = Chem.MolFromSmiles(smiles)
            smiles = Chem.MolToSmiles(mol, canonical=True) 
            if len(Chem.GetMolFrags(mol)) > 1:
                continue
            atom_mapping = mol.GetSubstructMatch(mol_from_sdf)
            pos = np.zeros_like(pos_from_sdf)
            for sdf_idx, smiles_idx in enumerate(atom_mapping):
                pos[smiles_idx] = pos_from_sdf[sdf_idx]
            num_atoms = mol.GetNumAtoms()

            
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

            data.pos = torch.from_numpy(pos).to(torch.float)

            # 定义要检查的目标行
            target_row = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float)
            # 检查 pos 中是否有与 [0, 0, 0] 相等的行
            is_zero_row = torch.all(torch.from_numpy(pos).to(torch.float) == target_row, dim=1)


            # data.energy = torch.Tensor([energy])
            # data.energy2 = torch.Tensor([energy2])
            espf_smiles, data.tokens, data.attention_mask, substructure_num, data.atom2substructure = espf_tokenize(smiles,mol)
            if espf_smiles is None:
                continue
            data.ori_smiles = smiles

            if data.pos.size()[0] == 0 or data.pos.size()[1] == 0:
                print("zero!")
                print(data.pos.size())
                continue
            data.num_nodes = num_atoms
            data.y = torch.Tensor(y[i].unsqueeze(0))


            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, "./dataset/3d/QM9/processed/data_alpha.pt")