from typing import Iterable, Optional, List
import pickle, os, uuid, shutil, tempfile
from sacred.run import Run
import matplotlib.pyplot as plt
from transformers import PreTrainedModel
from renard.pipeline.core import PipelineState
from renard.plot_utils import CharactersGraphLayout


def archive_pipeline_state_(_run: Run, state: PipelineState, name: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        pickle_path = f"{tmpdir}/{name}.pickle"
        with open(pickle_path, "wb") as f:
            pickle.dump(state, f)
        _run.add_artifact(pickle_path)


def archive_graph_(
    _run: Run, state: PipelineState, graph_name: str, layout: CharactersGraphLayout
):
    with tempfile.TemporaryDirectory() as tmpdir:
        # PNG export
        # ----------
        png_path = f"{tmpdir}/{graph_name}.png"
        fig = plt.gcf()
        fig.set_size_inches(24, 24)
        fig.set_dpi(300)
        state.plot_graph_to_file(png_path, fig=fig, layout=layout)
        _run.add_artifact(png_path)

        # Pickle export
        # -------------
        pickle_path = f"{tmpdir}/{graph_name}.pickle"
        with open(pickle_path, "wb") as f:
            pickle.dump(state.character_network, f)
        _run.add_artifact(pickle_path)

        # GEXF export
        # -----------
        # set graph coordinates for Gephi export
        for node in state.character_network.nodes:
            state.character_network.nodes[node]["viz"] = {
                "position": {
                    "x": layout[node][0] * 1000,
                    "y": layout[node][1] * 1000,
                    "z": 0.0,
                }
            }
        gexf_path = f"{tmpdir}/{graph_name}.gexf"
        state.export_graph_to_gexf(gexf_path)
        _run.add_artifact(gexf_path)


def log_series(
    _run: Run, name: str, series: Iterable, steps: Optional[List[int]] = None
):
    """Log the given 1D series to the given sacred run

    :param _run:
    :param name: metrics name
    :param series: series to log
    """
    for elt_i, elt in enumerate(series):
        step = steps[elt_i] if not steps is None else None
        _run.log_scalar(name, elt, step)


def archive_fig(_run: Run, fig, name: str, extension: str = "pdf"):
    """Archive a matplotlib figure

    :param _run: current sacred run
    :param fig: matplotlib figure
    :param name: name of the archived file, without extension
    :param extension:
    """
    tmp_name = f"{str(uuid.uuid4())}.{extension}"
    fig.savefig(tmp_name, dpi=300)

    _run.add_artifact(tmp_name, f"{name}.{extension}")

    plt.close(fig)
    os.remove(tmp_name)


def archive_text(_run: Run, text: str, name: str):
    """Archive a text as a text file.

    :param _run: current sacred run
    :param text: text to archive
    :param name: name of the output file, without extension
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/{name}.txt"
        with open(path, "w") as f:
            f.write(text)
        _run.add_artifact(path)


def archive_dir(
    run: Run, dir: str, dir_archive_name: Optional[str] = None, and_delete: bool = False
):
    """Archive a directory as a tar.gz archive

    :param run: current sacred run
    :param dir: directory to save
    :param dir_archive_name: name of the archive (format :
        ``f"{dir_archive_name}.tar.gz"``).  If ``None``, default to
        ``dir``.
    :param and_delete: if ``True``, ``dir`` will be deleted after
        archival.
    """
    if dir_archive_name is None:
        dir_archive_name = dir

    # archiving with shutil.make_archive somehow (somehow !) crashes
    # the sacred FileObserver. Maybe they monkeypatched something ?
    # anyway, here is an os.system hack. Interestingly, calling the
    # command directly is _way_ easier to figure out than using
    # shutil.make_archive. WTF python docs ?
    # /rant off
    os.system(f"tar -czvf {dir_archive_name}.tar.gz {dir}")
    run.add_artifact(f"{dir_archive_name}.tar.gz")

    # cleaning
    os.remove(f"./{dir_archive_name}.tar.gz")
    if and_delete:
        shutil.rmtree(dir)


def archive_huggingface_model(run: Run, model: PreTrainedModel, model_name: str):
    """Naive implementation of a huggingface model artifact saver

    :param run: current sacred run
    :param model: hugginface model to save
    :param model_name: name of the saved model
    """
    # surely no one will have a generated UUID as a filename... right?
    tmp_model_name = str(uuid.uuid4())
    model.save_pretrained(f"./{tmp_model_name}")
    archive_dir(run, tmp_model_name, dir_archive_name=model_name, and_delete=True)
