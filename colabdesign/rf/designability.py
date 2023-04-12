import os
from colabdesign.mpnn import mk_mpnn_model
from colabdesign.af import mk_af_model

# from colabdesign.shared.protein import pdb_to_string
# from colabdesign.shared.parse_args import parse_args
from rfdiffusion.inference.utils import parse_pdb
import rfdiffusion
from colabdesign.rf.utils import fix_contigs, fix_partial_contigs, fix_pdb

import subprocess
import os
import shlex
import pandas as pd
import numpy as np
from string import ascii_uppercase, ascii_lowercase
from pathlib import Path
import torch


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


RFDIFF_PATH = (Path(rfdiffusion.__file__) / "../../scripts/run_inference.py").resolve()
os.environ["DGLBACKEND"] = "pytorch"
alphabet_list = list(ascii_uppercase + ascii_lowercase)


def get_info(contig):
    F = []
    fixed_chain = True
    sub_contigs = [x.split("-") for x in contig.split("/")]
    for n, (a, b) in enumerate(sub_contigs):
        if a[0].isalpha():
            L = int(b) - int(a[1:]) + 1
            F += [1] * L
        else:
            L = int(b)
            F += [0] * L
            fixed_chain = False
    return F, fixed_chain


def convert_mpnn_priors_to_af_(priors):
    # reorder "mpnn -> colabdesign af"
    # returns bias and decoding order
    from colabdesign.mpnn.model import _aa_convert
    from protein_tools.util import get_priors
    from protein_tools.util import to_np

    weights = torch.from_numpy(priors)
    priors, entropy = get_priors(weights, return_entropy=True)
    decoding_order = to_np(torch.argsort(entropy, dim=-1))[:, None]
    priors_logits = (priors / (1 - priors)).log()
    priors_logits[priors_logits.isneginf()] = -1e11
    priors_logits[priors_logits.isinf()] = 1e11
    return _aa_convert(to_np(priors_logits), rev=True), decoding_order


def get_mpnn_bias(mutation, mutation_sites, seq, preserve_C=False):
    from protein_tools.proteinmpnn import alphabet

    # reorder "esm -> pmpnn"
    ignore_tokens = [x for x in mutation.alphabet_tokens if x not in alphabet]
    assert len(mutation.alphabet_tokens) - len(ignore_tokens) == len(alphabet)
    idxs = torch.tensor(
        [
            j
            for i, x in enumerate(alphabet)
            for j, y in enumerate(mutation.alphabet_tokens)
            if x == y
        ]
    )
    # fill positions other than mutation sites and the WT residues with zero
    esm_priors = np.zeros((len(seq), len(alphabet)))
    for s, p in zip(mutation_sites, mutation.residue_probs[0][:, idxs]):
        if preserve_C and seq[s] == "C":
            print("[Preserving the C in the WT sequence]")
            esm_priors[s, :] = 0.0
            esm_priors[s, alphabet.index("C")] = 1.0
        else:
            esm_priors[s, :] = p
    for idx, res in enumerate(seq):
        esm_priors[idx, alphabet.index(res)] = 1.0

    return convert_mpnn_priors_to_af_(esm_priors)


class RFdiff:
    def __init__(
        self,
        contigs,
        runid,
        pdb_filename,
        output="./outputs",
        RFDIFF_PATH=RFDIFF_PATH,
        iterations=50,
        symmetry="cyclic",
        copies=1,
        hotspot=None,
    ):
        # determine mode
        contigs = contigs.replace(",", " ").replace(":", " ").split()
        is_fixed, is_free = False, False
        for contig in contigs:
            for x in contig.split("/"):
                a = x.split("-")[0]
                if a[0].isalpha():
                    is_fixed = True
                if a.isnumeric():
                    is_free = True
        if len(contigs) == 0 or not is_free:
            mode = "partial"
        elif is_fixed:
            mode = "fixed"
        else:
            mode = "free"

        # fix input contigs
        if mode in ["partial", "fixed"]:
            parsed_pdb = parse_pdb(pdb_filename)
            opts = f" inference.input_pdb={pdb_filename}"
            if mode in ["partial"]:
                partial_T = int(80 * (iterations / 200))
                opts += f" diffuser.partial_T={partial_T}"
                contigs = fix_partial_contigs(contigs, parsed_pdb)
            else:
                opts += f" diffuser.T={iterations}"
                contigs = fix_contigs(contigs, parsed_pdb)
        else:
            opts = f" diffuser.T={iterations}"
            parsed_pdb = None
            contigs = fix_contigs(contigs, parsed_pdb)

        if hotspot is not None and hotspot != "":
            opts += f" ppi.hotspot_res=[{hotspot}]"

        # setup symmetry
        if copies > 1:
            sym = {"cyclic": "c", "dihedral": "d"}[symmetry] + str(copies)
            sym_opts = f"--config-name symmetry  inference.symmetry={sym} \
'potentials.guiding_potentials=[\"type:olig_contacts,weight_intra:1,weight_inter:0.1\"]' \
potentials.olig_intra_all=True potentials.olig_inter_all=True \
potentials.guide_scale=2 potentials.guide_decay=quadratic"
            opts = f"{sym_opts} {opts}"
            if symmetry == "dihedral":
                copies *= 2
            contigs = sum([contigs] * copies, [])

        opts = f"{opts} 'contigmap.contigs=[{' '.join(contigs)}]'"

        output_prefix = f"{output}/{runid}"
        print("mode:", mode)
        print("output:", f"{output_prefix}")
        print("contigs:", contigs)
        print()

        self.cmd = f"{RFDIFF_PATH} {opts} inference.output_prefix={output_prefix} inference.num_designs=1"

        self.runid = runid
        self.output_prefix = output_prefix
        self.output = output
        self.contigs = contigs
        self.copies = copies
        self.pdb_filename = pdb_filename
        self.done = False

    def run(self):
        print("running RFdiffusion...")
        print(self.cmd)
        process = subprocess.Popen(
            shlex.split(self.cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ,
        )
        stdout, stderr = process.communicate()
        if stderr != b"":
            print(stderr.decode())
            if 'Error' in stderr.decode():
                raise RuntimeError()
        else:
            self.post_run_fix()
        self.final_pdb = f"{self.output_prefix}_0.pdb"
        self.done = True

    def post_run_fix(self):
        # fix pdbs
        pdbs = [
            f"{self.output}/traj/{self.runid}_0_pX0_traj.pdb",
            f"{self.output}/traj/{self.runid}_0_Xt-1_traj.pdb",
            f"{self.output}/{self.runid}_0.pdb",
        ]
        for pdb in pdbs:
            with open(pdb, "r") as handle:
                pdb_str = handle.read()
            with open(pdb, "w") as handle:
                handle.write(fix_pdb(pdb_str, self.contigs))

    def visualise(
        self,
        show_mainchains=False,
        animate=True,
        color="chain",
        hbondCutoff=4.0,
    ):
        import py3Dmol
        from colabdesign.shared.plot import pymol_color_list
        from string import ascii_uppercase, ascii_lowercase

        alphabet_list = list(ascii_uppercase + ascii_lowercase)
        view = py3Dmol.view(js="https://3dmol.org/build/3Dmol.js")
        if animate:
            pdb = f"{self.output}/traj/{self.runid}_0_pX0_traj.pdb"
            pdb_str = open(pdb, "r").read()
            view.addModelsAsFrames(pdb_str, "pdb", {"hbondCutoff": hbondCutoff})
        else:
            pdb = self.final_pdb
            pdb_str = open(pdb, "r").read()
            view.addModel(pdb_str, "pdb", {"hbondCutoff": hbondCutoff})

        if color == "rainbow":
            view.setStyle({"cartoon": {"color": "spectrum"}})
        elif color == "chain":
            for n, chain, color in zip(
                range(len(self.contigs)), alphabet_list, pymol_color_list
            ):
                view.setStyle({"chain": chain}, {"cartoon": {"color": color}})
        view.zoomTo()
        if animate:
            view.animate({"loop": "backAndForth"})
        return view.show()


class Designability:
    def __init__(
        self,
        starting_seq=None,
        pdb=None,
        loc=None,
        contigs=None,
        copies=1,
        num_seqs=8,
        initial_guess=False,
        use_multimer=False,
        num_recycles=3,
    ):
        self.starting_seq = starting_seq
        self.pdb = pdb
        self.loc = loc
        self.contigs = contigs
        self.copies = copies
        self.num_seqs = num_seqs
        self.initial_guess = initial_guess
        self.use_multimer = use_multimer
        self.num_recycles = num_recycles
        if None in [self.starting_seq, self.pdb, self.loc, self.contigs]:
            raise RuntimeError("Missing Required Arguments")

        contigs = self.contigs.split(":")
        self.chains = alphabet_list[: len(contigs)]
        info = [get_info(x) for x in contigs]
        self.fixed_chains = [y for x, y in info]
        self.fixed_pos = sum([x for x, y in info], [])

        self.flags = {
            "initial_guess": self.initial_guess,
            "best_metric": "rmsd",
            "use_multimer": self.use_multimer,
            "model_names": [
                "model_1_multimer_v3" if self.use_multimer else "model_1_ptm"
            ],
        }

        self.fix_pos_flag = np.array(self.fixed_pos)
        assert self.fixed_chains == [False], "only single chain is supported!"
        self.motif_pos = np.nonzero(self.fix_pos_flag == 0)[0]

        # get esm_seq and check if C is in the loop
        cs = self.contigs.split("/")
        assert len(cs) == 3, "only single loop is supported"
        s, e, l = (
            int(cs[0].split("-")[-1]),
            int(cs[-1].split("-")[0][1:]) - 1,
            int(cs[1].split("-")[0]),
        )
        self.loop_has_cystine = "C" in starting_seq[s:e]
        self.esm_seq = starting_seq[: s + l // 2] + starting_seq[e - (l - l // 2) :]
        self.esm_starting_motif = starting_seq[s:e]
        if self.loop_has_cystine:
            print("Warning: loop has Cystine! Cystine is allowed to be sampled for this loop.")
        self.reset()

    def reset(self):
        self.out = None
        self.bias = None
        self.decoding_order = None
        self.af_model = None
        self.mpnn_model = None

    def init_af_mpnn(self, rm_aa):
        if sum(self.fixed_chains) > 0 and sum(self.fixed_chains) < len(
            self.fixed_chains
        ):
            self.protocol = "binder"
            print("protocol=binder")
            target_chains = []
            binder_chains = []
            for n, x in enumerate(self.fixed_chains):
                if x:
                    target_chains.append(self.chains[n])
                else:
                    binder_chains.append(self.chains[n])
            af_model = mk_af_model(protocol="binder", **self.flags)
            af_model.prep_inputs(
                self.pdb,
                target_chain=",".join(target_chains),
                binder_chain=",".join(binder_chains),
                rm_aa=rm_aa,
            )
        elif sum(self.fixed_pos) > 0:
            self.protocol = "partial"
            print("protocol=partial")
            af_model = mk_af_model(protocol="fixbb", use_templates=True, **self.flags)
            rm_template = np.array(self.fixed_pos) == 0
            af_model.prep_inputs(
                self.pdb,
                chain=",".join(self.chains),
                rm_template=rm_template,
                rm_template_seq=rm_template,
                copies=self.copies,
                homooligomer=self.copies > 1,
                rm_aa=rm_aa,
            )
            p = np.where(self.fixed_pos)[0]
            af_model.opt["fix_pos"] = p[p < af_model._len]

        else:
            self.protocol = "fixbb"
            print("protocol=fixbb")
            af_model = mk_af_model(protocol="fixbb", **self.flags)
            af_model.prep_inputs(
                self.pdb,
                chain=",".join(self.chains),
                copies=self.copies,
                homooligomer=self.copies > 1,
                rm_aa=rm_aa,
            )
        self.af_model = af_model
        self.mpnn_model = mk_mpnn_model()
        self.mpnn_model.get_af_inputs(self.af_model)

    def run_esm2(self, esm_model=None, esm_alphabet=None):
        print("running ESM2...")
        from protein_tools.mutation import ESMResidueMutation

        if esm_model is None or esm_alphabet is None:
            import esm
            esm_model, esm_alphabet = esm.pretrained.esm2_t36_3B_UR50D()

        esm_model = esm_model.to("cuda").eval()
        data = [(self.contigs, self.esm_seq)]
        self.mutation = (
            ESMResidueMutation(data, mutation_sites=self.motif_pos)
            .mask_data()
            .compute_logits(esm_model, esm_alphabet)
            .process_logits()
        )
        self.bias, self.decoding_order = get_mpnn_bias(
            self.mutation, self.motif_pos, self.esm_seq
        )

    def plot_esm_bias(self, ax=None, filename=None, **kwargs):
        from protein_tools.colabdesign.tools import af_alphabet

        if self.bias is None:
            raise RuntimeError("first run .run_esm()!")
        if ax is None:
            fig, ax = plt.subplots()
        ax = sns.heatmap(
            torch.from_numpy(self.bias[self.motif_pos, :]).softmax(-1).T,
            cmap="crest",
            linewidths=0.05,
            square=True,
            yticklabels=af_alphabet,
            xticklabels=self.motif_pos,
            ax=ax,
            **kwargs,
        )
        ax.set_title(
            f"{self.contigs}\n{self.esm_starting_motif}",
            fontsize="small",
            fontfamily="monospace",
        )
        if filename is not None:
            plt.savefig(filename)
        return ax

    def run_mpnn(self, mpnn_temp=0.1, bias_temp=1.0, **kwargs):
        if self.mpnn_model is None:
            self.init_af_mpnn(rm_aa=None if self.loop_has_cystine else "C")
        print("running proteinMPNN...")
        if self.bias is not None and "bias" not in kwargs:
            kwargs["bias"] = self.bias / bias_temp
            assert self.decoding_order is not None
            kwargs["decoding_order"] = self.decoding_order
            print("using the instance bias/decoding_order for hybrid sequence sampling")
        elif "bias" in kwargs and kwargs["bias"] is not None:
            kwargs["bias"] /= bias_temp
        if "decoding_order" in kwargs and kwargs["decoding_order"] is not None:
            self.decoding_order = kwargs["decoding_order"]

        n, cnt = 0, 1
        while n < self.num_seqs and cnt < 5:
            self.out = self.mpnn_model.sample(
                num=cnt * self.num_seqs // 8,
                batch=8,
                temperature=mpnn_temp,
                rescore=self.copies > 1,  # self.homooligomer,
                **kwargs,
            )
            n = self.remove_duplicates()
            cnt += 1

        mean = lambda item: np.mean(self.out[item])
        std = lambda item: np.std(self.out[item])
        k = "score"
        print(
            f"ProteinMPNN sampled {len(self.out['seq'])} (requested {(cnt-1)*self.num_seqs}) "
            f"unique sequences with <{k}>={mean(k):.2f}±{std(k):.2f}"
        )

        self.log_terms = []
        terms = ["score", "seq", "motif_seq", "original_motif_seq", "mpnn_temp"]
        self.out["mpnn_temp"] = [mpnn_temp] * len(self.out["seq"])
        if "bias" in kwargs:
            self.out["bias_temp"] = [bias_temp] * len(self.out["seq"])
            terms.append("bias_temp")

        self.out["motif_seq"] = [
            "".join(np.array(list(s))[self.motif_pos]) for s in self.out["seq"]
        ]
        self.out["original_motif_seq"] = [
            self.esm_starting_motif for s in self.out["seq"]
        ]
        self.log_terms.extend(terms)

    def remove_duplicates(self):
        _, idx = np.unique(self.out["seq"], return_index=True)
        for k in self.out:
            self.out[k] = self.out[k][idx, ...]
        return len(self.out["seq"])

    def run_af(self, all_path=None, prefix=''):
        if self.af_model is None:
            self.init_af_mpnn(rm_aa=None if self.loop_has_cystine else "C")
        if len(self.out["seq"]) == 0:
            raise RuntimeError("First run .run_mpnn() to sample sequences!")
        print("running AlphaFold", end="")
        if self.protocol == "binder":
            self.af_terms = ["plddt", "i_ptm", "i_pae", "rmsd"]
        elif self.copies > 1:
            self.af_terms = ["plddt", "ptm", "i_ptm", "pae", "i_pae", "rmsd"]
        else:
            self.af_terms = ["plddt", "motif_plddt", "ptm", "pae", "motif_pae", "rmsd"]
        for k in self.af_terms:
            self.out[k] = []
        self.out["pdb_path"] = []

        all_path = all_path or Path(f"{self.loc}/all_pdbs").resolve()
        all_path.mkdir(exist_ok=True, parents=True)
        with open(f"{self.loc}/design.fasta", "a") as fasta:
            for n in range(len(self.out["seq"])):
                seq = self.out["seq"][n][-self.af_model._len :]
                current_pdb = all_path / f"{prefix}{n}.pdb"
                self.out["pdb_path"].append(str(current_pdb))
                self.af_model.predict(
                    seq=seq, num_recycles=self.num_recycles, verbose=False
                )
                for t in self.af_terms:
                    if t == "motif_plddt":
                        self.out[t].append(
                            self.af_model.aux["plddt"][self.motif_pos].mean()
                        )
                    elif t == "motif_pae":
                        self.out[t].append(
                            self.af_model.aux["pae"][self.motif_pos, :].mean()
                        )
                    else:
                        self.out[t].append(self.af_model.aux["log"][t])
                if "i_pae" in self.out:
                    self.out["i_pae"][-1] = self.out["i_pae"][-1] * 31
                if "pae" in self.out:
                    self.out["pae"][-1] = self.out["pae"][-1] * 31
                self.af_model.save_current_pdb(f"{current_pdb}")
                self.af_model._save_results(save_best=True, verbose=False)
                self.af_model._k += 1
                score_line = [f'mpnn:{self.out["score"][n]:.3f}']
                for t in self.af_terms:
                    score_line.append(f"{t}:{self.out[t][n]:.3f}")
                # print(n, " ".join(score_line))# + " " + seq)
                print(".", end="")
                score_line.append(f"pdb_path:{current_pdb}")
                line = f'>{"|".join(score_line)}\n{seq}'
                fasta.write(line + "\n")
        print()
        self.best_rmsd_pdb = all_path / f"{prefix}_best_rmsd.pdb"
        self.af_model.save_pdb(f"{self.best_rmsd_pdb}")
        self.log_terms.extend(self.af_terms + ["pdb_path"])

    def run_RF2NA(self):
        if len(self.out["seq"]) == 0:
            raise RuntimeError("First run .run_mpnn() to sample sequences!")
        raise NotImplementedError

    def run_ESMIF(self):
        if len(self.out["seq"]) == 0:
            raise RuntimeError(
                "First run .run_mpnn() and .run_af() to sample sequences generate structures!"
            )
        raise NotImplementedError

    def generate_af_video(self, color_by="plddt"):
        assert color_by in ["chain", "plddt", "rainbow"]
        return self.af_model.animate(color_by=color_by)

    def generate_log(self, save=True):
        labels = self.log_terms.copy()
        if "seq" in labels:
            labels.remove("seq")
            labels.append("seq")
        data = [[self.out[k][n] for k in labels] for n in range(len(self.out["seq"]))]
        assert labels[0] == "score"
        labels[0] = "mpnn"  # rename score -> mpnn
        df = pd.DataFrame(data, columns=labels)
        if save:
            df.to_csv(f"{self.loc}/mpnn_results.csv", index=False)
        return df

    def visualise_pdb(self, pdb_path=None, hbondCutoff=4.0):
        import py3Dmol

        view = py3Dmol.view(js="https://3dmol.org/build/3Dmol.js")
        if pdb_path is None:
            pdb_path = f"{self.best_rmsd_pdb}"
            print(f"visualising '{pdb_path}'")
        pdb_str = open(f"{self.loc}_0.pdb", "r").read()
        view.addModel(pdb_str, "pdb", {"hbondCutoff": hbondCutoff})
        pdb_str = open(pdb_path, "r").read()
        view.addModel(pdb_str, "pdb", {"hbondCutoff": hbondCutoff})

        view.setStyle(
            {"model": 0}, {"cartoon": {}}
        )  #: {'colorscheme': {'prop':'b','gradient': 'roygb','min':0,'max':100}}})
        view.setStyle(
            {"model": 1},
            {
                "cartoon": {
                    "colorscheme": {
                        "prop": "b",
                        "gradient": "roygb",
                        "min": 0,
                        "max": 100,
                    }
                }
            },
        )
        view.zoomTo()
        return view.show()
