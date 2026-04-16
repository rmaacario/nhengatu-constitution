"""
conftest.py — shared pytest fixtures.
"""

import sys
from pathlib import Path

import pytest

# Make sure src/ is importable without pip install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corpus_pipeline.config import load, Config


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load()


# ---------------------------------------------------------------------------
# Minimal synthetic texts that mimic real PDF output
# ---------------------------------------------------------------------------

PT_SAMPLE = """\
PREÂMBULO

Nós, representantes do povo brasileiro, reunidos em Assembleia Nacional Constituinte
para instituir um Estado democrático.

TÍTULO I
Dos Princípios Fundamentais
Art. 1o A República Federativa do Brasil, formada pela união indissolúvel dos Estados
e Municípios e do Distrito Federal, constitui-se em Estado Democrático de Direito e
tem como fundamentos:
I – a soberania;
II – a cidadania;
Parágrafo único. Todo o poder emana do povo.

Art. 2o São Poderes da União, independentes e harmônicos entre si, o Legislativo, o
Executivo e o Judiciário.

Art. 3o Constituem objetivos fundamentais da República:
I – construir uma sociedade livre;
§ 1o Observado o disposto neste artigo.
§ 2o Vedada a adoção.

ATO DAS DISPOSIÇÕES
CONSTITUCIONAIS TRANSITÓRIAS
Art. 1o As eleições municipais …
"""

NHE_SAMPLE = """\
YUPIRUNGÁ RẼDEWÁ
Yandé, yaãmuyakũtasá miíra-ita uikúwa braziu upé yayumuatiri yepé yaátirisa wasú.

SESEWÁRA I
YUPIRŨGÁSA RETÉWA RESÉWAÁRA
Art. 1º Tetãma braziu, uyumuyãwa muatirisá istadu-ita asui tẽdawá-ita asui distritu
federau, umuyupirú míra-ita uikú rupiáára purãga:
I- Suberania;
II- Mírasá-itá;
Ũbeusá yepéyũtu. Kirῖbawasá uri míra-itá sui.

Art. 2º Uniãu kirῖbawasá -ita, yepéyũwa-ita asui suurí waitá uyumuyárisá yepewasú
asui umyãwa-ita ley.

Art. 3º Tetãma wasú urikú maduaisá kuáye:
I- umuyã miíra-itá purãgarã;
§ 1o Umaã seésé kuá.
§ 2o Ũbá upuderi.
"""


@pytest.fixture
def pt_sample() -> str:
    return PT_SAMPLE


@pytest.fixture
def nhe_sample() -> str:
    return NHE_SAMPLE
