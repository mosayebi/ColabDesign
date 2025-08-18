import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from colabdesign.shared.plot import plot_pseudo_3D, pymol_cmap, _np_kabsch
from string import ascii_uppercase, ascii_lowercase
import numpy as np
from colabdesign.shared.model import order_aa
from colabdesign.shared.utils import clear_mem
from torch.cuda import empty_cache
from pathlib import Path

alphabet_list = list(ascii_uppercase + ascii_lowercase)

def to_np(x):
   return x.detach().cpu().numpy()

def sym_it(coords, center, cyclic_symmetry_axis, reflection_axis=None):
    def rotation_matrix(axis, theta):
        axis = axis / np.linalg.norm(axis)
        a = np.cos(theta / 2)
        b, c, d = -axis * np.sin(theta / 2)
        return np.array(
            [
                [
                    a * a + b * b - c * c - d * d,
                    2 * (b * c - a * d),
                    2 * (b * d + a * c),
                ],
                [
                    2 * (b * c + a * d),
                    a * a + c * c - b * b - d * d,
                    2 * (c * d - a * b),
                ],
                [
                    2 * (b * d - a * c),
                    2 * (c * d + a * b),
                    a * a + d * d - b * b - c * c,
                ],
            ]
        )

    def align_axes(coords, source_axis, target_axis):
        rotation_axis = np.cross(source_axis, target_axis)
        rotation_angle = np.arccos(np.dot(source_axis, target_axis))
        rot_matrix = rotation_matrix(rotation_axis, rotation_angle)
        return np.dot(coords, rot_matrix)

    # Center the coordinates
    coords = coords - center

    # Align cyclic symmetry axis with Z-axis
    z_axis = np.array([0, 0, 1])
    coords = align_axes(coords, cyclic_symmetry_axis, z_axis)

    if reflection_axis is not None:
        # Align reflection axis with X-axis
        x_axis = np.array([1, 0, 0])
        coords = align_axes(coords, reflection_axis, x_axis)
    return coords


def fix_partial_contigs(contigs, parsed_pdb):
    INF = float("inf")

    # get unique chains
    chains = []
    for c, i in parsed_pdb["pdb_idx"]:
        if c not in chains:
            chains.append(c)

    # get observed positions and chains
    ok = []
    for contig in contigs:
        for x in contig.split("/"):
            if x[0].isalpha:
                C, x = x[0], x[1:]
                S, E = -INF, INF
                if x.startswith("-"):
                    E = int(x[1:])
                elif x.endswith("-"):
                    S = int(x[:-1])
                elif "-" in x:
                    (S, E) = (int(y) for y in x.split("-"))
                elif x.isnumeric():
                    S = E = int(x)
                for c, i in parsed_pdb["pdb_idx"]:
                    if c == C and i >= S and i <= E:
                        if [c, i] not in ok:
                            ok.append([c, i])

    # define new contigs
    new_contigs = []
    for C in chains:
        new_contig = []
        unseen = []
        seen = []
        for c, i in parsed_pdb["pdb_idx"]:
            if c == C:
                if [c, i] in ok:
                    L = len(unseen)
                    if L > 0:
                        new_contig.append(f"{L}-{L}")
                        unseen = []
                    seen.append([c, i])
                else:
                    L = len(seen)
                    if L > 0:
                        new_contig.append(f"{seen[0][0]}{seen[0][1]}-{seen[-1][1]}")
                        seen = []
                    unseen.append([c, i])
        L = len(unseen)
        if L > 0:
            new_contig.append(f"{L}-{L}")
        L = len(seen)
        if L > 0:
            new_contig.append(f"{seen[0][0]}{seen[0][1]}-{seen[-1][1]}")
        new_contigs.append("/".join(new_contig))
    fixed_contigs = list(set(new_contigs)) # manually remove duplicates in the case of symmetric contigs
    print("WARNNING! fixed contigs:\noriginal:\t{new_contigs}\nfixed:   \t{fixed_contigs}")
    return fixed_contigs


def fix_contigs(contigs, parsed_pdb):
    def fix_contig(contig):
        INF = float("inf")
        X = contig.split("/")
        Y = []
        for n, x in enumerate(X):
            if x[0].isalpha():
                C, x = x[0], x[1:]
                S, E = -INF, INF
                if x.startswith("-"):
                    E = int(x[1:])
                elif x.endswith("-"):
                    S = int(x[:-1])
                elif "-" in x:
                    (S, E) = (int(y) for y in x.split("-"))
                elif x.isnumeric():
                    S = E = int(x)
                new_x = ""
                c_, i_ = None, 0
                for c, i in parsed_pdb["pdb_idx"]:
                    if c == C and i >= S and i <= E:
                        if c_ is None:
                            new_x = f"{c}{i}"
                        else:
                            if c != c_ or i != i_ + 1:
                                new_x += f"-{i_}/{c}{i}"
                        c_, i_ = c, i
                Y.append(new_x + f"-{i_}")
            elif "-" in x:
                # sample length
                s, e = x.split("-")
                m = np.random.randint(int(s), int(e) + 1)
                Y.append(f"{m}-{m}")
            elif x.isnumeric() and x != "0":
                Y.append(f"{x}-{x}")
        return "/".join(Y)

    return [fix_contig(x) for x in contigs]


def fix_pdb(pdb_str, contigs):
    def get_range(contig):
        L_init = 1
        R = []
        sub_contigs = [x.split("-") for x in contig.split("/")]
        for n, (a, b) in enumerate(sub_contigs):
            if a[0].isalpha():
                if n > 0:
                    pa, pb = sub_contigs[n - 1]
                    if pa[0].isalpha() and a[0] == pa[0]:
                        L_init += int(a[1:]) - int(pb) - 1
                L = int(b) - int(a[1:]) + 1
            else:
                L = int(b)
            R += range(L_init, L_init + L)
            L_init += L
        return R

    contig_ranges = [get_range(x) for x in contigs]
    R, C = [], []
    for n, r in enumerate(contig_ranges):
        R += r
        C += [alphabet_list[n]] * len(r)

    pdb_out = []
    r_, c_, n = None, None, 0
    for line in pdb_str.split("\n"):
        if line[:4] == "ATOM":
            c = line[21:22]
            r = int(line[22 : 22 + 5])
            if r_ is None:
                r_ = r
            if c_ is None:
                c_ = c
            if r != r_ or c != c_:
                n += 1
                r_, c_ = r, c
            pdb_out.append("%s%s%4i%s" % (line[:21], C[n], R[n], line[26:]))
        if line[:5] == "MODEL" or line[:3] == "TER" or line[:6] == "ENDMDL":
            pdb_out.append(line)
            r_, c_, n = None, None, 0
    return "\n".join(pdb_out)


def get_ca(pdb_filename, get_bfact=False):
    xyz = []
    bfact = []
    for line in open(pdb_filename, "r"):
        line = line.rstrip()
        if line[:4] == "ATOM":
            atom = line[12 : 12 + 4].strip()
            if atom == "CA":
                x = float(line[30 : 30 + 8])
                y = float(line[38 : 38 + 8])
                z = float(line[46 : 46 + 8])
                xyz.append([x, y, z])
                if get_bfact:
                    b_factor = float(line[60 : 60 + 6].strip())
                    bfact.append(b_factor)
    if get_bfact:
        return np.array(xyz), np.array(bfact)
    else:
        return np.array(xyz)


def get_Ls(contigs):
    Ls = []
    for contig in contigs:
        L = 0
        for n, (a, b) in enumerate(x.split("-") for x in contig.split("/")):
            if a[0].isalpha():
                L += int(b) - int(a[1:]) + 1
            else:
                L += int(b)
        Ls.append(L)
    return Ls


def make_animation(pos, plddt=None, Ls=None, ref=0, line_w=2.0, dpi=100):
    if plddt is None:
        plddt = [None] * len(pos)

    # center inputs
    pos = pos - pos[ref, None].mean(1, keepdims=True)

    # align to best view
    best_view = _np_kabsch(pos[ref], pos[ref], return_v=True, use_jax=False)
    pos = np.asarray([p @ best_view for p in pos])

    fig, (ax1) = plt.subplots(1)
    fig.set_figwidth(5)
    fig.set_figheight(5)
    fig.set_dpi(dpi)

    xy_min = pos[..., :2].min() - 1
    xy_max = pos[..., :2].max() + 1
    z_min = None  # pos[...,-1].min() - 1
    z_max = None  # pos[...,-1].max() + 1

    for ax in [ax1]:
        ax.set_xlim(xy_min, xy_max)
        ax.set_ylim(xy_min, xy_max)
        ax.axis(False)

    ims = []
    for pos_, plddt_ in zip(pos, plddt):
        if plddt_ is None:
            if Ls is None:
                img = plot_pseudo_3D(
                    pos_, ax=ax1, line_w=line_w, zmin=z_min, zmax=z_max
                )
            else:
                c = np.concatenate([[n] * L for n, L in enumerate(Ls)])
                img = plot_pseudo_3D(
                    pos_,
                    c=c,
                    cmap=pymol_cmap,
                    cmin=0,
                    cmax=39,
                    line_w=line_w,
                    ax=ax1,
                    zmin=z_min,
                    zmax=z_max,
                )
        else:
            img = plot_pseudo_3D(
                pos_,
                c=plddt_,
                cmin=50,
                cmax=90,
                line_w=line_w,
                ax=ax1,
                zmin=z_min,
                zmax=z_max,
            )
        ims.append([img])

    ani = animation.ArtistAnimation(fig, ims, blit=True, interval=120)
    plt.close()
    return ani.to_html5_video()




def get_priors_and_entropy(weights: np.ndarray):
    """Returns normalised priors and their entropy.cAt columns where normalised priors are all zero entropy is set to -1."""
    assert np.all(weights >= 0)
    priors = np.where(
        weights.sum(-1, keepdims=True) > 0,
        weights / weights.sum(-1, keepdims=True),
        0.0,
    )
    log_priors = np.log(priors, where=priors > 0, out=np.zeros_like(priors))
    entropy = np.where(priors.sum(-1) > 0, (-priors * log_priors).sum(-1), -1.0)
    return priors, entropy


def _clip_p(p):
    eps = np.finfo(float).eps
    return np.clip(p, a_min=eps, a_max=1 - eps)


def get_bias_and_decoding_order(weights: np.ndarray):
    priors, entropy = get_priors_and_entropy(weights)
    decoding_order = np.argsort(entropy, axis=-1)[:, None]
    priors = _clip_p(priors)
    logits = np.log(priors / (1 - priors))
    return logits, decoding_order

def symmetrise_logits(logits:np.ndarray, copies=1):
    return np.tile(logits.reshape(copies, logits.shape[0]//copies,-1).mean(0), (copies,1))

def _convert_mutation_probs(mutation, target_alphabet):
    """convert probs from mutation object to correctly ordered priors, filling out positions other than mutations sites.
    returns esm_priors with shape [len(seq), len(target_alphabet)]
    """
    idxs_map = np.array([mutation.alphabet.get_idx(a) for a in target_alphabet])
    WT_seq = mutation.data[0][0][1] if mutation.is_msa else mutation.data[0][1]
    # fill positions other than mutation sites and the WT residues at fixed positions with zero
    esm_priors = np.zeros((len(WT_seq), len(target_alphabet)))
    for s, p in zip(mutation.mutation_sites, mutation.residue_probs[0][:, idxs_map]):
        esm_priors[s, :] = p
    for i, res in enumerate(WT_seq):
        if i not in mutation.mutation_sites:
            esm_priors[i, :] = 0
            esm_priors[i, target_alphabet.index(res)] = 1
    return esm_priors


def run_esm(esm_model, esm_alphabet, data, mutation_sites, chunksize=10, device="cuda", disable_tqdm=True):
    from protein_tools.mutation import ESMResidueMutation
    esm_model = esm_model.to(device).eval()
    mutation = (
        ESMResidueMutation(data, mutation_sites=mutation_sites)
        .mask_data(disable_tqdm=disable_tqdm)
        .compute_logits(esm_model, esm_alphabet, chunksize=chunksize, disable_tqdm=disable_tqdm)
        .process_logits()
    )
    return mutation


def get_esm2_bias_and_decoding_order(seq, fixed_pos, copies=1, device="cuda"):
    return get_esm_bias_and_decoding_order(seq, fixed_pos, copies=copies, device=device, model_name='esm2_t36_3B_UR50D')


def get_esm_bias_and_decoding_order(seq, fixed_pos, copies=1, device="cuda", model_name='esm2_t36_3B_UR50D', **kwargs):
    import esm
    print(f"running {model_name} ...")
    alphabet = list(order_aa.values())
    mutation_sites = np.where(np.array(fixed_pos) == 0)[0]
    assert len(fixed_pos) == len(seq)
    esm_model, esm_alphabet = getattr(esm.pretrained, model_name)()
    esm_priors = _convert_mutation_probs(
        run_esm(esm_model, esm_alphabet, [("seq", seq)], mutation_sites, device=device, **kwargs),
        alphabet,
    )
    del esm_model, esm
    empty_cache()
    clear_mem()
    logits, decoding_order = get_bias_and_decoding_order(esm_priors)
    return symmetrise_logits(logits, copies), decoding_order


def get_msa_transformer_bias_and_decoding_order(msa, fixed_pos, copies=1, device="cuda", verbose=True, **kwargs):
    import esm

    if verbose: print("running ESM MSA-Transformer...")
    alphabet = list(order_aa.values())
    mutation_sites = np.where(np.array(fixed_pos) == 0)[0]

    full_seq = msa[0][1]
    full_seq_len = len(full_seq)
    if full_seq_len > 1023:
      print(f'msa sequence length ({full_seq_len}) is too long, considering only the last 1023 residues')
      msa = [(n, s[-1023:]) for n,s in msa]
    assert len(fixed_pos) == full_seq_len
    offset = len(fixed_pos) - len(msa[0][1])
    if offset < 0:
      raise RuntimeError("incompatible msa!")
    mutation_sites = mutation_sites - offset
    assert np.sum(mutation_sites<0) == 0, 'at least one mutation site is in the truncated msa region'
    esm_model, esm_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    esm_priors = _convert_mutation_probs(
        run_esm(esm_model, esm_alphabet, [msa], mutation_sites, chunksize=1, device=device), alphabet, **kwargs
    )
    if offset > 0 :
      l, n = esm_priors.shape
      b0 = np.zeros((full_seq_len-l, n))
      for idx, res in enumerate(full_seq[:-1023]):
        b0[idx, alphabet.index(res)] = 1
      esm_priors = np.concatenate((b0, esm_priors))

    del esm_model, esm
    empty_cache()
    clear_mem()
    logits, decoding_order = get_bias_and_decoding_order(esm_priors)
    return symmetrise_logits(logits, copies), decoding_order


def get_frame2seq_bias_and_decoding_order(
    pdb_path,
    fixed_pos,
    copies=1,
    chain_id=None,
    device="cuda",
    target_alphabet=list(order_aa.values()),
):
    import torch
    from protein_tools.pdb import PDB
    from frame2seq import Frame2seqRunner
    from frame2seq.utils import residue_constants
    from frame2seq.utils.pdb2input import get_inference_inputs

    print("running frame2seq...")
    seqs = PDB.load(pdb_path).get_seqs()
    chain_ids = [chain_id] if chain_id is not None and isinstance(chain_id, str) else list(seqs.keys())
    if len(seqs) > 1:
        print(
            f"concatenating features from chain_id={chain_ids} (`frame2seq` supports single-chain structures)"
        )

    runner = Frame2seqRunner(device=device)
    fixed_positions = np.where(np.array(fixed_pos) != 0)[0]

    # concatenate features for multi-chain structures
    features = [[x.to(device) for x in get_inference_inputs(pdb_path, c)] for c in chain_ids]
    seq_mask, aatype, X = [torch.concat([f[i] for f in features], dim=1) for i in range(len(features[0]))]

    input_aatype_onehot = torch.zeros(tuple(aatype.shape)+(len(residue_constants.ID_TO_AA),), device=device)
    input_aatype_onehot[:, :, 20] = 1  # all positions are masked (set to unknown)
    for pos in fixed_positions:
        input_aatype_onehot[:, pos, :] = 0
        input_aatype_onehot[:, pos, aatype[0][pos]] = 1  # fixed positions set to the input sequence

    with torch.no_grad():
        pred_seq1 = runner.models[0].forward(X, seq_mask, input_aatype_onehot)
        pred_seq2 = runner.models[1].forward(X, seq_mask, input_aatype_onehot)
        pred_seq3 = runner.models[2].forward(X, seq_mask, input_aatype_onehot)
        logits = (pred_seq1 + pred_seq2 + pred_seq3) / 3  # ensemble
        entropy = - (logits.softmax(-1)*logits.log_softmax(-1)).sum(-1)
        decoding_order = entropy[0].argsort(-1)

    alphabet = list(residue_constants.ID_TO_AA.values())
    idxs_map = torch.tensor([alphabet.index(a) for a in target_alphabet])
    logits, decoding_order = to_np(logits[0, :, idxs_map]), to_np(decoding_order[:, None])
    return symmetrise_logits(logits, copies), decoding_order


def get_extra_aa_bias(aa_bias_dict, seq_len, bias=None):
    aa_bias = np.zeros((seq_len, len(order_aa)))
    aa_bias_dict = {aa.upper(): b for aa, b in aa_bias_dict.items()}
    for i, aa in order_aa.items():
        aa_bias[:, i] = aa_bias_dict.get(aa, 0)
    return aa_bias if bias is None else bias+aa_bias

def get_ligand_mpnn_logits_and_decoding_order(
    input_pdb,
    out_folder,
    args_str="--model_type ligand_mpnn --single_aa_score 1  --use_sequence 1  --batch_size 1 --number_of_batches 50",
    copies=1,
    score_py=None,
    target_alphabet=list(order_aa.values()),
):
    from ligandmpnn import ROOT
    import subprocess
    import torch

    print("running LigandMPNN...")

    if score_py is None:
        score_py = ROOT / "scripts/score.py"

    command = f"python {score_py}  --pdb_path {input_pdb} --out_folder {out_folder} {args_str}"
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    stdout, stderr = process.communicate()
    Path(out_folder).mkdir(exist_ok=True, parents=True)
    with open(f"{out_folder}/stderr", "wb") as f:
        f.write(stderr)
        if stderr != b"":
            print(stderr.decode())
    with open(f"{out_folder}/stdout", "wb") as f:
        f.write(stdout)
    with open(f"{out_folder}/command", "w") as f:
        f.write(command)

    name = Path(f"{input_pdb}").stem
    path = Path(f"{out_folder}/{name}.pt")
    x = torch.load(path)
    alphabet = x["alphabet"]
    logits = torch.from_numpy(x["logits"].mean(0))
    entropy = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)
    decoding_order = entropy.argsort(-1)
    idxs_map = torch.tensor([alphabet.index(a) for a in target_alphabet])
    logits, decoding_order = to_np(logits[:, idxs_map]), to_np(decoding_order[:, None])
    return symmetrise_logits(logits, copies), decoding_order


def get_saprot_bias_and_decoding_order(
    pdb_path,
    fixed_pos,
    copies=1,
    device="cuda",
    chain_id=None,
    target_alphabet=list(order_aa.values()),
):

    print("running SaProt...")
    import saprot
    from saprot.utils.foldseek_util import get_struc_seq
    from saprot.utils.constants import aa_list
    from saprot.model.saprot.saprot_foldseek_mutation_model import SaprotFoldseekMutationModel
    from protein_tools.pdb import PDB

    seqs = PDB.load(pdb_path).get_seqs()
    chain_ids = [chain_id] if chain_id is not None and isinstance(chain_id, str) else list(seqs.keys())

    DIR = Path(saprot.__file__).parent.parent
    config = {
    "foldseek_path": DIR / "bin/foldseek",
    "config_path": DIR / "weights/PLMs/SaProt_650M_AF2",
    "load_pretrained": True,
    }

    parsed_seqs = get_struc_seq(
        config['foldseek_path'],
        pdb_path, chain_ids
    )

    if len(parsed_seqs)>1:
        print('PDB has more than one chain. concatenating the sequence ...')
    seq = ''.join([x[0] for x in parsed_seqs.values()])
    combined_seq = ''.join([x[-1] for x in parsed_seqs.values()])
    print(f"SaProt combined sequence:\n{combined_seq}\n")

    model = SaprotFoldseekMutationModel(**config)
    model.eval()
    model.to(device)
    priors = np.stack([list(model.predict_pos_prob(combined_seq, resid).values()) for resid in range(1,1+len(seq))])
    logits, decoding_order = get_bias_and_decoding_order(priors)
    idxs_map = np.array([aa_list.index(a) for a in target_alphabet])
    logits = logits[:, idxs_map]
    return symmetrise_logits(logits, copies), decoding_order