from pathlib import Path
import argparse
import os
from datetime import datetime
import time, signal
import random, string
import numpy as np
import colabdesign

# from google.colab import files
import json
import matplotlib.pyplot as plt
import ipywidgets as widgets
from protein_tools.pdb import PDB, VisualiseStructure
from IPython.display import display, HTML
import py3Dmol

from rfdiffusion.inference.utils import parse_pdb
from colabdesign.rf.utils import get_ca
from colabdesign.rf.utils import fix_contigs, fix_partial_contigs, fix_pdb, sym_it
from colabdesign.shared.protein import pdb_to_string
from colabdesign.shared.plot import plot_pseudo_3D
import rfdiffusion

os.environ["DGLBACKEND"] = "pytorch"
RFDIFF_ROOT = Path(rfdiffusion.__file__).parent.parent
DESIGNABILITY_SCRIPT = Path(colabdesign.__file__).parent / "rf/designability_test.py"
assert DESIGNABILITY_SCRIPT.exists()


def create_view(
    path,
    color_scheme="chainindex",
    select=lambda x, y: y.id == "A",
    spin=True,
    **kwargs,
):
    p = PDB.load(path)
    if color_scheme == "bfactor":
        view = (
            VisualiseStructure(p)
            .color_by(
                color_scheme, opacity=1.0, color_domain=[0, 100], color_reverse=False
            )
            .show()
        )
    else:
        view = (
            VisualiseStructure(p)
            .add_fixed_structure(
                p.select(select),
                color_scheme=color_scheme,
                color="red",
                opacity=1.0,
            )
            .color_by(color_scheme, opacity=0.4, **kwargs)
        ).show()
    if spin:
        view.control.spin([1, 0, 0], angle=np.pi / 2)
    view.handle_resize()
    view.center()
    return view


def create_label(label, width="90%", text_align="center", font_size="10px"):
    return widgets.Label(
        value=label,
        layout=widgets.Layout(width=width, text_align=text_align, font_size=font_size),
    )


def view_pore(
    pdb_path,
    ref_pdb_path=None,
    ref_chains=["A"],
    label="",
    select=lambda x, y: y.id == "A",
    **kwargs,
):
    items = [
        create_label(label),
        widgets.HBox(
            [
                create_view(
                    pdb_path,
                    color_scheme="chainindex",
                    select=select,
                    spin=np.pi / 2,
                    **kwargs,
                ),
                create_view(
                    pdb_path, color_scheme="bfactor", select=select, spin=np.pi / 2
                ),
                create_view(pdb_path, color_scheme="bfactor", spin=None),
            ]
        ),
    ]
    if ref_pdb_path:
        items.append(
            widgets.HBox(
                [
                    create_view(
                        ref_pdb_path,
                        color_scheme="bfactor",
                        select=select,
                        spin=np.pi / 2,
                    ),
                    VisualiseStructure(
                        PDB.load(ref_pdb_path).select(lambda x, y: y.id in ref_chains)
                    )
                    .TMalign_with(
                        PDB.load(pdb_path).select(lambda x, y: y.id in ref_chains),
                        verbose=False,
                    )
                    .color_by(
                        color_scheme="bfactor",
                        color_domain=[0, 100],
                        color_reverse=False,
                    )
                    .show(),
                ]
            )
        )

    rows, cols = 1, 1
    grid = widgets.GridBox(
        [widgets.VBox(items)],
        layout=widgets.Layout(
            grid_template_columns=f"repeat({cols:0}, 1fr)",
            grid_template_rows=f"repeat({rows:0}, 1fr)",
            grid_gap="0px",
        ),
    )
    return grid


# def view_pore(pdb_file, label="", select=lambda x, y: y.id == "A", **kwargs):
#     rows, cols = 1, 1
#     grid = widgets.GridBox(
#         [
#             widgets.VBox(
#                 [
#                     create_label(label),
#                     widgets.HBox(
#                         [
#                             create_view(
#                                 pdb_file,
#                                 color_scheme="chainindex",
#                                 select=select,
#                                 spin=True,
#                                 **kwargs,
#                             ),
#                             create_view(pdb_file, color_scheme="bfactor", spin=False),
#                         ]
#                     ),
#                 ]
#             )
#         ],
#         layout=widgets.Layout(
#             grid_template_columns=f"repeat({cols:0}, 1fr)",
#             grid_template_rows=f"repeat({rows:0}, 1fr)",
#             grid_gap="0px",
#         ),
#     )
#     return grid


def get_pdb(pdb_code):
    if pdb_code is None or pdb_code == "":
        raise RuntimeError()
    #         upload_dict = files.upload()
    #         pdb_string = upload_dict[list(upload_dict.keys())[0]]
    #         with open("tmp.pdb", "wb") as out:
    #             out.write(pdb_string)
    #         return "tmp.pdb"
    elif os.path.isfile(pdb_code):
        return pdb_code
    elif len(pdb_code) == 4:
        if not os.path.isfile(f"{pdb_code}.pdb1"):
            os.system(f"wget -qnc https://files.rcsb.org/download/{pdb_code}.pdb1.gz")
            os.system(f"gunzip {pdb_code}.pdb1.gz")
        return f"{pdb_code}.pdb1"
    else:
        os.system(
            f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v3.pdb"
        )
        return f"AF-{pdb_code}-F1-model_v3.pdb"


def run_ananas(pdb_str, path, sym=None):
    pdb_filename = f"outputs/{path}/ananas_input.pdb"
    out_filename = f"outputs/{path}/ananas.json"
    with open(pdb_filename, "w") as handle:
        handle.write(pdb_str)

    cmd = f"ananas {pdb_filename} -u -j {out_filename}"
    if sym is None:
        os.system(cmd)
    else:
        os.system(f"{cmd} {sym}")

    # parse results
    try:
        out = json.loads(open(out_filename, "r").read())
        results, AU = out[0], out[-1]["AU"]
        group = AU["group"]
        chains = AU["chain names"]
        rmsd = results["Average_RMSD"]
        print(f"AnAnaS detected {group} symmetry at RMSD:{rmsd:.3}")

        C = np.array(results["transforms"][0]["CENTER"])
        A = [np.array(t["AXIS"]) for t in results["transforms"]]

        # apply symmetry and filter to the asymmetric unit
        new_lines = []
        for line in pdb_str.split("\n"):
            if line.startswith("ATOM"):
                chain = line[21:22]
                if chain in chains:
                    x = np.array([float(line[i : (i + 8)]) for i in [30, 38, 46]])
                    if group[0] == "c":
                        x = sym_it(x, C, A[0])
                    if group[0] == "d":
                        x = sym_it(x, C, A[1], A[0])
                    coord_str = "".join(["{:8.3f}".format(a) for a in x])
                    new_lines.append(line[:30] + coord_str + line[54:])
            else:
                new_lines.append(line)
        return results, "\n".join(new_lines)

    except:
        return None, pdb_str


def run(command, steps, num_designs=1, visual="none"):
    def run_command_and_get_pid(command):
        pid_file = "/dev/shm/pid"
        print(f"RFdiffusion command:\n{command}\n", flush=True)
        os.system(f"nohup {command} & echo $! > {pid_file}")
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())
        os.remove(pid_file)
        return pid

    def is_process_running(pid):
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        else:
            return True

    run_output = widgets.Output()
    progress = widgets.FloatProgress(
        min=0, max=1, description="running", bar_style="info"
    )
    display(widgets.VBox([progress, run_output]))

    # clear previous run
    for n in range(steps):
        if os.path.isfile(f"/dev/shm/{n}.pdb"):
            os.remove(f"/dev/shm/{n}.pdb")

    pid = run_command_and_get_pid(command)
    try:
        fail = False
        for _ in range(num_designs):
            # for each step check if output generated
            for n in range(steps):
                wait = True
                while wait and not fail:
                    time.sleep(0.1)
                    if os.path.isfile(f"/dev/shm/{n}.pdb"):
                        pdb_str = open(f"/dev/shm/{n}.pdb").read()
                        if pdb_str[-3:] == "TER":
                            wait = False
                        elif not is_process_running(pid):
                            fail = True
                    elif not is_process_running(pid):
                        fail = True

                if fail:
                    progress.bar_style = "danger"
                    progress.description = "failed"
                    break

                else:
                    progress.value = (n + 1) / steps
                    if visual != "none":
                        with run_output:
                            run_output.clear_output(wait=True)
                            if visual == "image":
                                xyz, bfact = get_ca(f"/dev/shm/{n}.pdb", get_bfact=True)
                                fig = plt.figure()
                                fig.set_dpi(100)
                                fig.set_figwidth(6)
                                fig.set_figheight(6)
                                ax1 = fig.add_subplot(111)
                                ax1.set_xticks([])
                                ax1.set_yticks([])
                                plot_pseudo_3D(xyz, c=bfact, cmin=0.5, cmax=0.9, ax=ax1)
                                plt.show()
                            if visual == "interactive":
                                view = py3Dmol.view(
                                    js="https://3dmol.org/build/3Dmol.js"
                                )
                                view.addModel(pdb_str, "pdb")
                                view.setStyle(
                                    {
                                        "cartoon": {
                                            "colorscheme": {
                                                "prop": "b",
                                                "gradient": "roygb",
                                                "min": 0.5,
                                                "max": 0.9,
                                            }
                                        }
                                    }
                                )
                                view.zoomTo()
                                view.show()
                if os.path.exists(f"/dev/shm/{n}.pdb"):
                    os.remove(f"/dev/shm/{n}.pdb")
            if fail:
                progress.bar_style = "danger"
                progress.description = "failed"
                break

        while is_process_running(pid):
            time.sleep(0.1)

    except KeyboardInterrupt:
        os.kill(pid, signal.SIGTERM)
        progress.bar_style = "danger"
        progress.description = "stopped"


def run_diffusion(
    contigs,
    path,
    pdb=None,
    iterations=50,
    symmetry="none",
    order=1,
    hotspot=None,
    chains=None,
    add_potential=False,
    potential_string=["type:olig_contacts,weight_intra:1,weight_inter:0.1"],
    num_designs=1,
    visual="none",
    extra_rfdiff_opts=[],
):
    full_path = f"outputs/{path}"
    os.makedirs(full_path, exist_ok=True)
    opts = [
        f"inference.output_prefix={full_path}",
        f"inference.num_designs={num_designs}",
    ]

    if chains == "":
        chains = None

    # determine symmetry type
    if symmetry in ["auto", "cyclic", "dihedral"]:
        if symmetry == "auto":
            sym, copies = None, 1
        else:
            sym, copies = {
                "cyclic": (f"c{order}", order),
                "dihedral": (f"d{order}", order * 2),
            }[symmetry]
    else:
        symmetry = None
        sym, copies = None, 1

    # determine mode
    contigs = contigs.replace(",", " ").replace(":", " ").split()
    is_fixed, is_free = False, False
    fixed_chains = []
    for contig in contigs:
        for x in contig.split("/"):
            a = x.split("-")[0]
            if a[0].isalpha():
                is_fixed = True
                if a[0] not in fixed_chains:
                    fixed_chains.append(a[0])
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
        pdb_str = pdb_to_string(get_pdb(pdb), chains=chains)
        if symmetry == "auto":
            a, pdb_str = run_ananas(pdb_str, path)
            if a is None:
                print(f"ERROR: no symmetry detected")
                symmetry = None
                sym, copies = None, 1
            else:
                if a["group"][0] == "c":
                    symmetry = "cyclic"
                    sym, copies = a["group"], int(a["group"][1:])
                elif a["group"][0] == "d":
                    symmetry = "dihedral"
                    sym, copies = a["group"], 2 * int(a["group"][1:])
                else:
                    print(
                        f'ERROR: the detected symmetry ({a["group"]}) not currently supported'
                    )
                    symmetry = None
                    sym, copies = None, 1

        elif mode == "fixed":
            pdb_str = pdb_to_string(pdb_str, chains=fixed_chains)

        pdb_filename = f"{full_path}/input.pdb"
        with open(pdb_filename, "w") as handle:
            handle.write(pdb_str)

        parsed_pdb = parse_pdb(pdb_filename)
        opts.append(f"inference.input_pdb={pdb_filename}")
        if mode in ["partial"]:
            iterations = int(80 * (iterations / 200))
            opts.append(f"diffuser.partial_T={iterations}")
            contigs = fix_partial_contigs(contigs, parsed_pdb)
        else:
            opts.append(f"diffuser.T={iterations}")
            contigs = fix_contigs(contigs, parsed_pdb)
    else:
        opts.append(f"diffuser.T={iterations}")
        parsed_pdb = None
        contigs = fix_contigs(contigs, parsed_pdb)

    if hotspot is not None and hotspot != "":
        opts.append(f"ppi.hotspot_res=[{hotspot}]")
        opts.append("denoiser.noise_scale_ca=0")
        opts.append("denoiser.noise_scale_frame=0")

    # setup symmetry
    if sym is not None:
        sym_opts = ["--config-name symmetry", f"inference.symmetry={sym}"]
        opts = sym_opts + opts
        contigs = sum([contigs] * copies, [])

    if add_potential:
        potential_string = r'","'.join(potential_string)
        opts += [
            f"'potentials.guiding_potentials=[\"{potential_string}\"]'",
            "potentials.olig_intra_all=True",
            "potentials.olig_inter_all=True",
            "potentials.guide_scale=2",
            "potentials.guide_decay=quadratic",
        ]

    opts.append(f"'contigmap.contigs=[{' '.join(contigs)}]'")
    opts += [
        f"inference.dump_pdb={visual !='none'}",
        "inference.dump_pdb_path='/dev/shm'",
    ]
    opts += extra_rfdiff_opts

    print("mode:", mode)
    print("output:", full_path)
    print("contigs:", contigs)

    opts_str = " ".join(opts)
    cmd = f"{RFDIFF_ROOT}/scripts/run_inference.py {opts_str}"
    # print(cmd, flush=True)

    # RUN
    run(cmd, iterations, num_designs, visual=visual)

    # fix pdbs
    for n in range(num_designs):
        pdbs = [
            f"outputs/traj/{path}_{n}_pX0_traj.pdb",
            f"outputs/traj/{path}_{n}_Xt-1_traj.pdb",
            f"{full_path}_{n}.pdb",
        ]
        for pdb in pdbs:
            with open(pdb, "r") as handle:
                pdb_str = handle.read()
            with open(pdb, "w") as handle:
                handle.write(fix_pdb(pdb_str, contigs))

    print(contigs)
    return contigs, copies


def run_sampling(args):
    for i in range(args.num_designs):
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        # @title run **RFdiffusion** to generate a backbone
        name = f"{args.prefix}_{i}"  # @param {type:"string"}
        contigs = args.contigs  # @param {type:"string"}
        pdb = str(args.pdb)  # @param {type:"string"}
        iterations = (
            args.iterations
        )  # @param ["25", "50", "100", "150", "200"] {type:"raw"}
        hotspot = args.hotspot  # @param {type:"string"}
        num_designs = 1  # @param ["1", "2", "4", "8", "16", "32"] {type:"raw"}
        visual = args.visual  # @param ["none", "image", "interactive"]
        # @markdown ---
        # @markdown **symmetry** settings
        # @markdown ---
        symmetry = args.symmetry  # @param ["none", "auto", "cyclic", "dihedral"]
        order = (
            args.order
        )  # @param ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"] {type:"raw"}
        chains = args.chains  # @param {type:"string"}
        add_potential = not (args.potential_string is None or args.potential_string == [""] or args.potential_string == [])
        # @param {type:"boolean"}
        # @markdown - `symmetry='auto'` enables automatic symmetry dectection with [AnAnaS](https://team.inria.fr/nano-d/software/ananas/).
        # @markdown - `chains="A,B"` filter PDB input to these chains (may help auto-symm detector)
        # @markdown - `add_potential` to discourage clashes between chains

        # determine where to save
        path = name
        while os.path.exists(f"outputs/{path}_0.pdb"):
            path = (
                name
                + "_"
                + "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
            )

        flags = {
            "contigs": contigs,
            "pdb": pdb,
            "order": order,
            "iterations": iterations,
            "symmetry": symmetry,
            "hotspot": hotspot,
            "path": path,
            "chains": chains,
            "add_potential": add_potential,
            "potential_string": args.potential_string,
            "num_designs": num_designs,
            "visual": visual,
            "extra_rfdiff_opts": args.extra_rfdiff_opts,
        }
        print(flags)
        for k, v in flags.items():
            if isinstance(v, str):
                flags[k] = v.replace("'", "").replace('"', "")

        contigs, copies = run_diffusion(**flags)

        print(f"RFdiffusion done ({name})!")
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 100)

        # @title run **ProteinMPNN** to generate a sequence and **AlphaFold** to validate
        num_seqs = (
            args.num_sequences_per_design
        )  # @param ["1", "2", "4", "8", "16", "32", "64"] {type:"raw"}
        initial_guess = args.initial_guess  # @param {type:"boolean"}
        num_recycles = (
            args.num_recycles
        )  # @param ["0", "1", "2", "3", "6", "12"] {type:"raw"}
        use_multimer = args.use_multimer  # @param {type:"boolean"}
        rm_aa = args.rm_aa  # @param {type:"string"}
        # @markdown - for **binder** design, we recommend `initial_guess=True num_recycles=3`

        contigs_str = ":".join(contigs)
        opts = [
            f"--pdb=outputs/{path}_0.pdb",
            f"--loc=outputs/{path}",
            f"--contig={contigs_str}",
            f"--copies={copies}",
            f"--num_seqs={num_seqs}",
            f"--num_recycles={num_recycles}",
            f"--rm_aa={rm_aa}",
            f"--num_designs={num_designs}",
        ]
        opts = opts + [f'--{x}' for x in args.extra_designability_opts]
        if initial_guess:
            opts.append("--initial_guess")
        if use_multimer:
            opts.append("--use_multimer")
        if copies > 1:
            assert use_multimer

        print("=" * 100)
        print(f"running designability_test ...")
        opts = f"python {DESIGNABILITY_SCRIPT} " + " ".join(opts)
        print(f"designability_test command:\n{opts}\n", flush=True)
        os.system(opts)
        print(f"designability test done ({name})!")
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 100 + "\n" * 2)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", type=Path, default=None, help="The input pdb file.")
    parser.add_argument(
        "--contigs",
        type=str,
        required=True,
        help="`contigs` are used to define continuous chains. "
        "Use a `:` to define multiple contigs and a `/` to define multiple segments within a contig. "
        "Example: --contigs 'A1-120/2-22/B4-20/2-20/A125-182'",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="design",
        help="The `prefix` gets added to output file names. Note that all output files are generated in the `./outputs` directory. Default prefix is 'design'.",
    )
    parser.add_argument(
        "--num_designs",
        type=int,
        default=10,
        help="Number of RFdiffusion designs (Default is 10)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        choices=[25, 50, 100, 150, 200],
        help="Number of RFdiffusion denoising steps (Default is 50).",
    )
    parser.add_argument(
        "--hotspot", type=str, default="", help="Hotspot residues for binder design"
    )
    parser.add_argument(
        "--symmetry",
        type=str,
        default="none",
        choices=["none", "auto", "cyclic", "dihedral"],
        help="Symmetry of the structure. Use 'auto' to infer the symmetry automatically using `ananas` and 'none' for disabling the symmetric design. "
        "Default is 'none'.",
    )
    parser.add_argument(
        "--order", type=int, default=1, help="The symmetry order. Default is 1."
    )
    parser.add_argument(
        "--chains",
        type=str,
        default="",
        help="Filter PDB input to these chains (may help auto-symm detector). Default is ''.",
    )
    parser.add_argument(
        "--potential_string",
        type=str,
        nargs="+",
        default=[],
        help="Guiding potential in the RFdiffusion inference. To use more than one potential, leave a space between each potential string. By default no potential is applied. "
        "Example: --potential_string 'type:olig_contacts,weight_intra:1,weight_inter:0.1'",
    )
    parser.add_argument(
        "--num_sequences_per_design",
        type=int,
        default=32,
        choices=[1, 2, 4, 8, 16, 32, 64],
        help="Number of MPNN sequences per RFdiffusion design (default is 32).",
    )
    parser.add_argument(
        "--initial_guess",
        action="store_true",
        default=True,
        help="Uses the input PDB coordinates as an initial guess in AlphaFold prediction (similar to the coordinate input from previous recycle). "
        "Note that this is a weak restraint, as the model was trained in the regime where coordinates form previous recycle maybe incorrect. "
        "While `initial_guess` is a weak restraint, using templates is considered a strong one. For templates, the strength of the template restraint is "
        "proportional to identity between the query sequence and the template sequence.",
    )
    parser.add_argument(
        "--num_recycles",
        type=int,
        default=3,
        help="Number of AlphaFold recycles (default is 3).",
    )
    parser.add_argument(
        "--use_multimer",
        action="store_true",
        default=False,
        help="To use the AlphaFold multimer model.",
    )
    parser.add_argument(
        "--rm_aa",
        type=str,
        default="C",
        help="The amino acids to avoid in the MPNN sequence sampling step (default is 'C')",
    )
    parser.add_argument(
        "--visual",
        type=str,
        default="none",
        choices=["none", "image", "interactive"],
        help="The visualisation mode of the intermediate structures during RFdiffusion inference, useful when the script is executed in a notebook.",
    )

    parser.add_argument(
        "--extra_rfdiff_opts",
        type=str,
        nargs="+",
        default=[],
        help="extra command line options to be passed in to the rfdiffusion inference script",
    )

    parser.add_argument(
        "--extra_designability_opts",
        type=str,
        nargs="+",
        default=[],
        help="extra command line options to be passed in to the designability_test.py script."
        "As an example, for sampling with ESM2+MPNN at mppn_temp=0.2 and bias_temp=1.0 set\n"
        "--extra_designability_opts esm2_priors  bias_temp=1.0 mpnn_temp=0.2"
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print()
    run_sampling(args)
