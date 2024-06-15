import glob
import pathlib as pl
from typing import List, Tuple
from sacred import Ingredient
from rich.progress import track
from flatten_litbank_ner import flatten_litbank_ner_
from splice.data import Novel, load_novel

litbank_ingredient = Ingredient("litbank")


@litbank_ingredient.config
def litbank_config():
    root: str
    flat_ner_output_dir: str = "./flat_litbank_ner"
    keep_only_NER_mentions: bool = False


@litbank_ingredient.capture
def load_litbank(
    root: str, flat_ner_output_dir: str, keep_only_NER_mentions: bool
) -> List[Novel]:

    flatten_litbank_ner_(root, flat_ner_output_dir)

    litbank_plpath = pl.Path(root).expanduser()
    coref_path = litbank_plpath / "coref" / "conll"

    ner_paths = [pl.Path(path) for path in glob.glob(f"{flat_ner_output_dir}/*.conll")]
    coref_paths = [coref_path / ner_path.name for ner_path in ner_paths]

    novels = [
        load_novel(ner_path, coref_path, keep_only_NER_mentions=keep_only_NER_mentions)
        for ner_path, coref_path in track(
            zip(ner_paths, coref_paths), total=len(ner_paths)
        )
    ]

    return novels
