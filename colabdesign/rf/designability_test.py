import os, sys

from colabdesign.mpnn import mk_mpnn_model
from colabdesign.af import mk_af_model
from colabdesign.shared.protein import pdb_to_string
from colabdesign.shared.parse_args import parse_args

from pathlib import Path
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
from string import ascii_uppercase, ascii_lowercase

alphabet_list = list(ascii_uppercase + ascii_lowercase)

from protein_tools.util import save_obj, load_obj
from protein_tools.pdb import PDB
from colabdesign.rf.utils import (
    get_esm2_bias_and_decoding_order,
    get_msa_transformer_bias_and_decoding_order,
    get_frame2seq_bias_and_decoding_order,
    get_extra_aa_bias,
    get_ligand_mpnn_logits_and_decoding_order,
    get_saprot_bias_and_decoding_order,
    get_e1_bias_and_decoding_order,
    sample_from_msa,
)


def get_info(contig):
    F = []
    free_chain = False
    fixed_chain = False
    sub_contigs = [x.split("-") for x in contig.split("/")]
    for n, (a, b) in enumerate(sub_contigs):
        if a[0].isalpha():
            L = int(b) - int(a[1:]) + 1
            F += [1] * L
            fixed_chain = True
        else:
            L = int(b)
            F += [0] * L
            free_chain = True
    return F, [fixed_chain, free_chain]


def main(argv):
    ligand_mpnn_args = f"--model_type ligand_mpnn --single_aa_score 1  --use_sequence 1  --batch_size 1 --number_of_batches 50"

    ag = parse_args()
    ag.txt("-------------------------------------------------------------------------------------")
    ag.txt("Designability Test")
    ag.txt("-------------------------------------------------------------------------------------")
    ag.txt("REQUIRED")
    ag.txt( "-------------------------------------------------------------------------------------")
    ag.add(["pdb="], None, str, ["input pdb"])
    ag.add(["loc="], None, str, ["location to save results"])
    ag.add(["contigs="], None, str, ["contig definition"])
    ag.txt("-------------------------------------------------------------------------------------")
    ag.txt("OPTIONAL")
    ag.txt("-------------------------------------------------------------------------------------")
    ag.add(["copies="], 1, int, ["number of repeating copies"])
    ag.add(["num_seqs="], 8, int, ["number of mpnn designs to evaluate"])
    ag.add(["initial_guess"], False, None, ["initialize previous coordinates"])
    ag.add(["use_multimer"], False, None, ["use alphafold_multimer_v3"])
    ag.add(["use_soluble"], False, None, ["use solubleMPNN"])
    ag.add(["num_recycles="], 3, int, ["number of recycles"])
    ag.add(["rm_aa="], "C", str, ["disable specific amino acids from being sampled"])
    ag.add(["num_designs="], 1, int, ["number of designs to evaluate"])
    ag.add(["mpnn_temp="], 0.1, float, ["sampling temperature used by proteinMPNN"])
    ag.add(["bias_temp="], 1.0, float, ["sampling temperature used to scale bias"])
    ag.add(["ligand_mpnn_temp="], 0.1, float, ["sampling temperature used to scale ligand_mpnn logits"],)
    ag.add(["ligand_mpnn"], False, None, ["enables ligand_mpnn priors"])
    ag.add(["ligand_mpnn_pdb="], False, str, ["input pdb file with ligands for the ligand_mpnn model"],)
    ag.add(["esm2_bias"], False, None, ["enables ESM2 priors"])
    ag.add(["input_seq="], None, str, ["input sequence for ESM2 priors calculation, if not given the sequence is taken from the input PDB"],)
    ag.add(["msa_transformer_bias"], False, None, ["enables ESM MSA Transformer priors"])
    ag.add(["input_msa="], None, str, ["input msa file for MSA-Transformer priors calculation (format: .a3m)"],)
    ag.add(["max_msa_depth="], 500, int, ["max msa depth, default is 500"])
    ag.add(["frame2seq_bias"], False, None, ["enables Frame2seq priors"])
    ag.add(["saprot_bias"], False, None, ["enables SaProt priors"])
    ag.add(["e1_bias"], False, None, ["enables E1 priors"])
    ag.add(["e1_context_max_tokens="], 60000, int, ["max tokens for the E1 MSA context. Final bias is the average of E1 logits over three progressively shorter contexts. default: 60000"])
    ag.add(["e1_num_seeds="], 1, int, ["number of seeds per context size when sampling E1 contexts"])
    ag.add(["e1_model_name="], 'E1-300M', str, ["E1 model name. default: 'E1-300M'"])
    ag.add(["bias_npy="], None, str, ["bias numpy array file"])
    ag.add(["decoding_order_npy="], None, str, ["decoding_order numpy array file"])
    ag.add(["aa_bias="], None, str, ["user-defined aa biases as a dict. e.g. \"{'A': -1.1, 'K': 0.7}\""],)
    ag.add(["save_logits"], False, None, ["save logits"])
    ag.add(["ligand_mpnn_args="], ligand_mpnn_args, str, ["(ligand_mpnn) score.py arguments. default: `{ligand_mpnn_args}`"],)
    ag.add(["fasta="], None, str, ["fasta file containing sequences to be assessed"])

    ag.txt("-------------------------------------------------------------------------------------")
    o = ag.parse(argv)

    def write_csv(data, labels, all_pdb_paths, verbose=False):
        labels_copy = labels.copy()
        labels_copy[2] = "mpnn"
        df = pd.DataFrame(data, columns=labels_copy).assign(
            input_pdb_path=str(Path(o.pdb).resolve()), **o.__dict__
        )
        df["pdb_path"] = all_pdb_paths
        csv = f"{o.loc}/designability_test_results.csv"
        df.drop("pdb", axis=1).to_csv(csv, index=False)
        if verbose:
            print(f"designability test results are saved to '{csv}'.")

    if None in [o.pdb, o.loc, o.contigs]:
        ag.usage("Missing Required Arguments")

    if o.rm_aa == "":
        o.rm_aa = None

    # filter contig input
    contigs = []
    for contig_str in o.contigs.replace(" ", ":").replace(",", ":").split(":"):
        if len(contig_str) > 0:
            contig = []
            for x in contig_str.split("/"):
                if x != "0":
                    contig.append(x)
            contigs.append("/".join(contig))

    chains = alphabet_list[: len(contigs)]
    info = [get_info(x) for x in contigs]
    fixed_pos = []
    fixed_chains = []
    free_chains = []
    both_chains = []
    for pos, (fixed_chain, free_chain) in info:
        fixed_pos += pos
        fixed_chains += [fixed_chain and not free_chain]
        free_chains += [free_chain and not fixed_chain]
        both_chains += [fixed_chain and free_chain]

    flags = {
        "initial_guess": o.initial_guess,
        "best_metric": "rmsd",
        "use_multimer": o.use_multimer,
        "model_names": ["model_1_multimer_v3" if o.use_multimer else "model_1_ptm"],
    }

    # work out biases
    bias_info, lmpnn_info = [], []
    for m in range(o.num_designs):
        bias, decoding_order = {}, {}
        if o.num_designs == 0:
            pdb_filename = o.pdb
        else:
            pdb_filename = o.pdb.replace("_0.pdb", f"_{m}.pdb")

        if o.ligand_mpnn:
            bias0, decoding_order0 = get_ligand_mpnn_logits_and_decoding_order(
                input_pdb=o.ligand_mpnn_pdb,
                out_folder=f"{o.loc}/ligand_mpnn/{m}",
                args_str=o.ligand_mpnn_args,
                copies=o.copies,
            )
            lmpnn_info.append(bias0.copy())
            decoding_order.update(
                {"ligand_mpnn_decoding_order": decoding_order0.copy()}
            )
        else:
            lmpnn_info.append(None)

        if o.bias_npy is not None:
            print(f"using bias from '{o.bias_npy}'")
            bias.update({"bias_npy": np.load(o.bias_npy)})

        if o.frame2seq_bias:
            bias0, decoding_order0 = get_frame2seq_bias_and_decoding_order(
                pdb_filename, fixed_pos, copies=o.copies, device="cuda"
            )  # does not contain rm_aa biases
            bias.update({"frame2seq_bias": bias0.copy()})
            decoding_order.update({"frame2seq_decoding_order": decoding_order0.copy()})

        if o.saprot_bias:
            bias0, decoding_order0 = get_saprot_bias_and_decoding_order(
                pdb_filename, fixed_pos, copies=o.copies, device="cuda"
            )  # does not contain rm_aa biases
            bias.update({"saprot_bias": bias0.copy()})
            decoding_order.update({"saprot_decoding_order": decoding_order0.copy()})

        if o.esm2_bias:
            if o.input_seq:
                seq = o.input_seq
            else:
                seq = "".join(PDB.load(pdb_filename).get_seqs().values())
                print(
                    f"input sequence for ESM2 priors inferred from the pdb file:\n{seq}"
                )
            bias0, decoding_order0 = get_esm2_bias_and_decoding_order(
                seq, fixed_pos, copies=o.copies, device="cuda"
            )  # does not contain rm_aa biases
            bias.update({"esm2_bias": bias0.copy()})
            decoding_order.update({"esm2_decoding_order": decoding_order0.copy()})

        if o.msa_transformer_bias:
            from protein_tools.msa import parse_a3m

            msa = parse_a3m(o.input_msa)
            if len(msa) > o.max_msa_depth:
                print(
                    f"input_msa is too deep. only considering the first max_msa_depth={o.max_msa_depth} sequences..."
                )
                msa = msa[: o.max_msa_depth]
            bias0, decoding_order0 = get_msa_transformer_bias_and_decoding_order(
                msa, fixed_pos, copies=o.copies, device="cuda"
            )  # does not contain rm_aa biases
            bias.update({"msa_transformer_bias": bias0.copy()})
            decoding_order.update(
                {"msa_transformer_decoding_order": decoding_order0.copy()}
            )

        if o.e1_bias:
            if o.input_seq:
                seq = o.input_seq
            else:
                seq = "".join(PDB.load(pdb_filename).get_seqs().values())
                print(
                    f"input sequence for E1 priors inferred from the pdb file:\n{seq}"
                )
            if o.input_msa:
              print(f"averaging bias over 3 contexts sampled with variable lengths from the input msa ...")
              ls_ = []
              for _ in range(3):
                for _s in range(o.e1_num_seeds):
                  context = sample_from_msa(o.input_msa, max_token_length=o.e1_context_max_tokens/(3-_))
                  print(f"[{_}_{_s}] max_tokens={o.e1_context_max_tokens/(3-_)}: sampled context contains {context.count(',')} sequences.")
                  bias0, decoding_order0 = get_e1_bias_and_decoding_order(
                    seq, fixed_pos, context=context, copies=o.copies, model_name=o.e1_model_name,
                  )
                  ls_.append(bias0.copy())
              bias0 = np.mean(ls_, axis=0)
            else:
              bias0, decoding_order0 = get_e1_bias_and_decoding_order(
                  seq, fixed_pos, context=None, copies=o.copies, model_name=o.e1_model_name,
                )
            bias.update({"e1_bias": bias0.copy()})
            decoding_order.update({"e1_decoding_order": decoding_order0.copy()})


        if o.aa_bias is not None:
            aa_bias_dict = eval(o.aa_bias)
            print(f"aa biases are loaded as {aa_bias_dict}")
            seq_len = len("".join(PDB.load(pdb_filename).get_seqs().values()))
            bias0 = get_extra_aa_bias(aa_bias_dict, seq_len, bias=None)
            bias.update({"aa_bias": bias0.copy()})

        if o.decoding_order_npy is not None:
            print(f"decoding_order loaded from '{o.decoding_order_npy}'")
            decoding_order = decoding_order.update(
                {"decoding_order_npy": np.load(o.decoding_order_npy)}
            )

        if len(bias) == 0:
            tot_bias = None
        elif len(bias) == 1:
            tot_bias = next(iter(bias.values()))
        else:
            print(f"summing up {list(bias.keys())} biases")
            tot_bias = sum(bias.values())

        tot_decoding_order = None
        for k in [
            "decoding_order_npy",
            "msa_transformer_decoding_order",
            "esm2_decoding_order",
            "frame2seq_decoding_order",
            "ligand_mpnn_decoding_order",
            "saprot_decoding_order",
            "e1_decoding_order",
        ]:
            if k in decoding_order:
                tot_decoding_order = decoding_order[k]
                print(f"decoding order is set to '{k}'")
                break
        bias_info.append((tot_bias, tot_decoding_order, bias, decoding_order))

    if sum(both_chains) == 0 and sum(fixed_chains) > 0 and sum(free_chains) > 0:
        protocol = "binder"
        print("protocol=binder")
        target_chains = []
        binder_chains = []
        for n, x in enumerate(fixed_chains):
            if x:
                target_chains.append(chains[n])
            else:
                binder_chains.append(chains[n])
        af_model = mk_af_model(protocol="binder", **flags)
        prep_flags = {
            "target_chain": ",".join(target_chains),
            "binder_chain": ",".join(binder_chains),
            "rm_aa": o.rm_aa,
        }
        opt_extra = {}

    elif sum(fixed_pos) > 0:
        protocol = "partial"
        print("protocol=partial")
        af_model = mk_af_model(protocol="fixbb", use_templates=True, **flags)
        rm_template = np.array(fixed_pos) == 0
        prep_flags = {
            "chain": ",".join(chains),
            "rm_template": rm_template,
            "rm_template_seq": rm_template,
            "copies": o.copies,
            "homooligomer": o.copies > 1,
            "rm_aa": o.rm_aa,
        }

    else:
        protocol = "fixbb"
        print("protocol=fixbb")
        af_model = mk_af_model(protocol="fixbb", **flags)
        prep_flags = {
            "chain": ",".join(chains),
            "copies": o.copies,
            "homooligomer": o.copies > 1,
            "rm_aa": o.rm_aa,
        }

    batch_size = 8
    if o.num_seqs < batch_size:
        batch_size = o.num_seqs

    if o.fasta is None:
        print("running proteinMPNN...")
        mpnn_model = mk_mpnn_model(
            weights="soluble" if o.use_soluble else "original", verbose=True
        )
        outs = []
        pdbs = []
        for m in range(o.num_designs):
            if o.num_designs == 0:
                pdb_filename = o.pdb
            else:
                pdb_filename = o.pdb.replace("_0.pdb", f"_{m}.pdb")
            pdbs.append(pdb_filename)
            af_model.prep_inputs(pdb_filename, **prep_flags)
            if protocol == "partial":
                p = np.where(fixed_pos)[0]
                af_model.opt["fix_pos"] = p[p < af_model._len]

            mpnn_model.get_af_inputs(af_model)
            bias, decoding_order, bias_terms, decoding_order_terms = bias_info[m]
            lmpnn_logits = lmpnn_info[m]
            sampling_kws = {}
            msg = f"mpnn_temp={o.mpnn_temp}"
            total_bias = (
                mpnn_model._inputs["bias"].copy() * o.mpnn_temp
            )  # to include rm_aa/fixed_pos biases in ._input['bias']
            if not lmpnn_logits is None:
                msg += f", ligand_mpnn_temp={o.ligand_mpnn_temp}"
                total_bias += lmpnn_logits * o.mpnn_temp / o.ligand_mpnn_temp
                sampling_kws.update({"bias": total_bias.copy()})
            if not bias is None:
                msg += f", bias_temp={o.bias_temp}"
                total_bias += bias * o.mpnn_temp / o.bias_temp
                sampling_kws.update({"bias": total_bias.copy()})
            if not decoding_order is None:
                sampling_kws.update({"decoding_order": decoding_order.copy()})
            print(f"[{Path(pdb_filename).stem}] sampling at {msg}")
            outs.append(
                mpnn_model.sample(
                    num=o.num_seqs // batch_size,
                    batch=batch_size,
                    temperature=o.mpnn_temp,
                    **sampling_kws,
                )
            )

            if o.save_logits:
                Path(o.loc).mkdir(exist_ok=True, parents=True)
                fn = f"{o.loc}/logits_{m}.pkl"
                save_obj(
                    dict(
                        mpnn_logits=mpnn_model.get_logits(),
                        mpnn_unconditional_logits=mpnn_model.get_unconditional_logits(),
                        total_bias=total_bias,
                        bias=bias,
                        ligand_mpnn_logits=lmpnn_logits,
                        decoding_order=decoding_order,
                        bias_terms=bias_terms,
                        decoding_order_terms=decoding_order_terms,
                        mutation_sites=np.where(np.array(fixed_pos) == 0)[0],
                        kwargs=o,
                    ),
                    fn,
                )
                print(f"logits are saved to '{fn}'.")
    else:
        print(f"sequences are loaded from `{o.fasta}`")
        raise NotImplementedError("--fasta is not implemented!")
        # read fasta
        from Bio import SeqIO

        fasta_sequences = SeqIO.parse(open(o.fasta), "fasta")
        outs = []
        pdbs = []
        for m in range(o.num_designs):
            if o.num_designs == 0:
                pdb_filename = o.pdb
            else:
                pdb_filename = o.pdb.replace("_0.pdb", f"_{m}.pdb")
            pdbs.append(pdb_filename)
            names, seqs = [], []
            for i in range(o.num_seqs):
                f = next(fasta_sequences)
                seqs.append(str(f.seq))
                names.append(f.id)
            outs.append({"name": names, "seq": seqs, "score": [np.nan] * o.num_seqs})

    if protocol == "binder":
        af_terms = ["plddt", "i_ptm", "i_pae", "rmsd"]
    elif o.copies > 1:
        af_terms = ["plddt", "ptm", "i_ptm", "pae", "i_pae", "rmsd"]
    else:
        af_terms = ["plddt", "ptm", "pae", "rmsd"]

    labels = ["design", "n", "score"] + af_terms + ["seq"]
    data = []
    best = {"rmsd": np.inf, "design": 0, "n": 0}
    print("running AlphaFold...")
    os.system(f"mkdir -p {o.loc}/all_pdb")
    all_pdb_paths = []
    with open(f"{o.loc}/design.fasta", "w") as fasta:
        for m, (out, pdb_filename) in enumerate(zip(outs, pdbs)):
            out["design"] = []
            out["n"] = []
            af_model.prep_inputs(pdb_filename, **prep_flags)
            for k in af_terms:
                out[k] = []
            for n in tqdm(range(o.num_seqs), desc=f"AF [{Path(pdb_filename).stem}]"):
                out["design"].append(m)
                out["n"].append(n)
                sub_seq = out["seq"][n].replace("/", "")[-af_model._len :]
                af_model.predict(
                    seq=sub_seq, num_recycles=o.num_recycles, verbose=False
                )
                for t in af_terms:
                    out[t].append(af_model.aux["log"][t])
                if "i_pae" in out:
                    out["i_pae"][-1] = out["i_pae"][-1] * 31
                if "pae" in out:
                    out["pae"][-1] = out["pae"][-1] * 31
                rmsd = out["rmsd"][-1]
                if rmsd < best["rmsd"]:
                    best = {"design": m, "n": n, "rmsd": rmsd}
                save_path = f"{o.loc}/all_pdb/design{m}_n{n}.pdb"
                af_model.save_current_pdb(save_path)
                all_pdb_paths.append(str(Path(save_path).resolve()))
                af_model._save_results(save_best=True, verbose=False)
                af_model._k += 1
                score_line = [f"design:{m} n:{n}", f'mpnn:{out["score"][n]:.3f}']
                for t in af_terms:
                    score_line.append(f"{t}:{out[t][n]:.3f}")
                print(" ".join(score_line) + " " + out["seq"][n])
                line = f'>{"|".join(score_line)}\n{out["seq"][n]}'
                fasta.write(line + "\n")
                data.append([out[k][n] for k in labels])
                write_csv(data, labels, all_pdb_paths, verbose=False)
            # data += [[out[k][n] for k in labels] for n in range(o.num_seqs)]
            af_model.save_pdb(f"{o.loc}/best_design{m}.pdb")

    # save best
    with open(f"{o.loc}/best.pdb", "w") as handle:
        remark_text = f"design {best['design']} N {best['n']} RMSD {best['rmsd']:.3f}"
        handle.write(f"REMARK 001 {remark_text}\n")
        handle.write(open(f"{o.loc}/best_design{best['design']}.pdb", "r").read())

    write_csv(data, labels, all_pdb_paths, verbose=True)


if __name__ == "__main__":
    main(sys.argv[1:])
