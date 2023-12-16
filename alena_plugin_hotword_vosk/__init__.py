from typing import List
from os.path import join, exists
import json
import enum
from speech_recognition import AudioData
from pathlib import Path
from vosk import KaldiRecognizer, Model as KaldiModel

from alena_plugin_manager.core.hotword_engine import HotWordEngine

# from alena_core.modules.client.speech.hotword_engine import HotWordEngine
from alena_core.utils.file import download_extract_zip, download_extract_tar
from alena_core.utils.fuzzy_matching import fuzzy_match, MatchStrategy
from alena_core.utils import LOG


def xdg_data_home() -> Path:
    """Возвращает путь XDG_DATA_HOME."""
    return Path.home() / ".local" / "share"


class MatchRule(str, enum.Enum):
    CONTAINS = "contains"
    EQUALS = "equals"
    STARTS = "starts"
    ENDS = "ends"
    FUZZY = "fuzzy"
    TOKEN_SET_RATIO = "token_set_ratio"
    TOKEN_SORT_RATIO = "token_sort_ratio"
    PARTIAL_TOKEN_SET_RATIO = "partial_token_set_ratio"
    PARTIAL_TOKEN_SORT_RATIO = "partial_token_sort_ratio"


class ModelContainer:
    """Класс для загрузки и хранения моделей kaldi."""

    UNK = "[unk]"
    SAMPLE_RATE = 16000

    def __init__(
        self, samples: List[str] = None, full_vocab: bool = False
    ) -> None:
        if not full_vocab and not samples:
            full_vocab = True

        samples = samples or []
        if self.UNK not in samples:
            samples.append(self.UNK)
        self.samples = samples
        self.full_vocab = full_vocab
        self.engine = None

    def get_engine(self, lang: str = None) -> KaldiRecognizer:
        """Возвращает модель для языка, загружает при необходимости.

        Args:
            lang (str, optional): Язык. None по умолчанию.

        Returns:
            _type_: движок для языка
        """
        if not self.engine and lang:
            lang = lang.split("-")[0].lower()
            self.load_language(lang)
        return self.engine

    def get_partial_transcription(self, lang: str = None) -> str:
        """Возвращает частичную транскрипцию.

        Args:
            lang (str, optional): язык модели. None по умолчанию.

        Returns:
            str: частичная транскрипция
        """
        engine = self.get_engine(lang)
        res = engine.PartialResult()
        return json.loads(res)["partial"]

    def get_final_transcription(self, lang: str = None) -> str:
        """Возвращает полную транскрипцию записи.

        Args:
            lang (str, optional): язык модели. None по умолчанию.

        Returns:
            str: транскрипция записи
        """
        engine = self.get_engine(lang)
        res = engine.FinalResult()
        return json.loads(res)["text"]

    def process_audio(self, audio: AudioData, lang: str = None):
        """Обрабатывает аудиозапись.

        Args:
            audio (AudioData): аудиозапись
            lang (str, optional): язык модели. None по умолчанию.

        Returns:
            str: транскрипция записи
        """
        engine = self.get_engine(lang)
        if isinstance(audio, AudioData):
            audio = audio.get_wav_data()
        return engine.AcceptWaveform(audio)

    def get_model(
        self, model_path: str, samples: List[str] = None, lang: str = None
    ) -> KaldiRecognizer:
        """Создает модель.

        Использует либо полный словарь, либо заданные ключевые фразы.

        Args:
            model_path (str): путь до файла с моделью kaldi
            samples (List[str], optional): ключевые фразы. None по умолчанию.
            lang (str): язык модели. None по умолчанию.

        Raises:
            FileNotFoundError: в случае, если не задан путь до модели

        Returns:
            KaldiRecognizer: модель kaldi
        """
        if model_path:
            if self.full_vocab:
                model = KaldiRecognizer(
                    KaldiModel(model_path, lang=lang), self.SAMPLE_RATE
                )
            else:
                model = KaldiRecognizer(
                    KaldiModel(model_path, lang=lang),
                    self.SAMPLE_RATE,
                    json.dumps(samples or self.samples, ensure_ascii=False)
                    .encode("utf8")
                    .decode(),
                )
            return model
        else:
            raise FileNotFoundError

    def load_model(self, model_path: str) -> None:
        """Загружает модель.

        Args:
            model_path (str): путь до файла с моделью.
        """
        self.engine = self.get_model(model_path, self.samples)

    def load_language(self, lang: str) -> None:
        """Загружает модель для заданного языка.

        Args:
            lang (str): язык модели
        """
        lang = lang.split("-")[0].lower()
        model_path = self.download_language(lang)
        self.load_model(model_path)

    @staticmethod
    def download_language(lang: str) -> str | None:
        """Загружает модель для заданного языка и возвращает
        путь к файлу с моделью или None, если язык не поддерживается.

        Args:
            lang (str): язык модели

        Returns:
            str: путь до локально загруженного файла с моделью
        """
        lang = lang.split("-")[0].lower()
        model_path = ModelContainer.lang2modelurl(lang)
        if model_path and model_path.startswith("http"):
            model_path = ModelContainer.download_model(model_path)
        return model_path

    @staticmethod
    def download_model(url: str) -> str:
        """Загружает модель vosk в указанную директорию.

        Args:
            url (str): ссылка для скачивания модели

        Returns:
            str: путь до локально скачанной модели
        """
        folder = join(xdg_data_home(), "vosk")
        name = url.split("/")[-1].split(".")[0]
        model_path = join(folder, name)
        if not exists(model_path):
            LOG.info(f"Загрузка модели vosk из {url}")
            LOG.info("Требуется немного времени...")
            if url.endswith(".zip"):
                download_extract_zip(url, folder=folder, folder_name=name)
            else:
                download_extract_tar(url, folder=folder, folder_name=name)
            LOG.info(f"Модель загружена в {model_path}")

        return model_path

    @staticmethod
    def lang2modelurl(lang: str, small: bool = True) -> str | None:
        """Возвращает ссылку для скачивания модели.

        Args:
            lang (str): язык модели
            small (bool, optional): True для малой модели. True по умолчанию.

        Returns:
            str | None: ссылка для скачивания или None,
                        если язык не поддерживается
        """
        lang2url = {
            "en": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",  # noqa: E501
            "ru": "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip",  # noqa: E501
        }
        biglang2url = {
            "en": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",  # noqa: E501
            "ru": "https://alphacephei.com/vosk/models/vosk-model-ru-0.22.zip",  # noqa: E501
        }
        if not small:
            lang2url.update(biglang2url)
        lang = lang.lower()
        if lang in lang2url:
            return lang2url[lang]
        lang = lang.split("-")[0]
        return lang2url.get(lang)


class VoskHotword(HotWordEngine):
    """Движок распознавания ключевой фразы на основе Vosk."""

    # Значения, жестко заданные в  alena/modules/client/speech/mic.py
    SEC_BETWEEN_WW_CHECKS = 0.2

    # максимальная продолжительность каждого аудиофрагмента
    MAX_EXPECTED_DURATION = 3

    def __init__(
        self, key_phrase: str = "алёна", config: dict = None, lang="ru-RU"
    ) -> None:
        config = config or {}
        super().__init__(key_phrase, config, lang)
        default_sample = [key_phrase.replace("_", " ").replace("-", " ")]
        self.full_vocab: bool = self.config.get("full_vocab", False)
        self.samples: List[str] = self.config.get("samples", default_sample)
        self.rule: MatchRule = self.config.get("rule", MatchRule.EQUALS)
        self.thresh = self.config.get("threshold", 0.75)
        self.debug = self.config.get("debug", False)
        self.time_between_checks = min(
            self.config.get("time_between_checks", 1.0), 3
        )
        self.expected_duration = self.MAX_EXPECTED_DURATION
        self._counter = 0
        self._load_model()

        if self.debug:
            LOG.debug("========= Настройки плагина VoskHotword ========= ")
            LOG.debug(f"    full_vocab: {self.full_vocab}")
            LOG.debug(f"    rule: {self.rule}")
            LOG.debug(f"    samples: {self.samples}")
            LOG.debug(f"    model: {self.config.get('model')}")
            LOG.debug(f"    time_between_checks: {self.time_between_checks}")
            LOG.debug("")

    def _load_model(self) -> None:
        """Загружает языковую модель Vosk."""
        model_path = self.config.get("model")
        self.model = ModelContainer(self.samples, self.full_vocab)
        if model_path:
            if model_path.startswith("http"):
                model_path = ModelContainer.download_model(model_path)
            self.model.load_model(model_path)
        else:
            self.model.load_language(self.lang)

    def found_wake_word(self, frame_data: bytes) -> bool:
        self._counter += self.SEC_BETWEEN_WW_CHECKS

        if self._counter < self.time_between_checks:
            return False
        self._counter = 0

        try:
            self.model.process_audio(frame_data, self.lang)
            transcript = self.model.get_final_transcription(self.lang)
        except Exception as e:
            LOG.error(f"Ошибка при обработке аудиозаписи: {repr(e)}")
            return False

        if not transcript or transcript == self.model.UNK:
            return False

        if self.debug:
            LOG.debug("Распознано kaldi: " + transcript)

        return self.apply_rules(
            transcript, self.samples, self.rule, self.thresh
        )

    @classmethod
    def apply_rules(
        cls,
        transcript: str,
        samples: List[str],
        rule: MatchRule = MatchRule.FUZZY,
        thresh: float = 0.75,
    ) -> bool:
        """Применяет выбранные правила на оценку соответствия произнесенной
        и транскрибированной фразы искомым примерам.

        Args:
            transcript (str): транскрипция произнесенной пользователем фразы
            samples (List[str]): ключевые фразы
            rule (MatchRule, optional): правило сопоставления.
                                MatchRule.FUZZY по умолчанию.
            thresh (float, optional): порог для нечеткого сопоставления.
                                0.75 по умолчанию.

        Returns:
            bool: соответствует ли транскрипция какому-то из
                    ключевых слов (фраз)
        """
        for s in samples:
            s = s.lower().strip()

            match rule:
                case MatchRule.FUZZY:
                    if fuzzy_match(s, transcript) >= thresh:
                        return True

                case MatchRule.TOKEN_SORT_RATIO:
                    if (
                        fuzzy_match(
                            s,
                            transcript,
                            strategy=MatchStrategy.TOKEN_SORT_RATIO,
                        )
                        >= thresh
                    ):
                        return True

                case MatchRule.TOKEN_SET_RATIO:
                    if (
                        fuzzy_match(
                            s,
                            transcript,
                            strategy=MatchStrategy.TOKEN_SET_RATIO,
                        )
                        >= thresh
                    ):
                        return True

                case MatchRule.PARTIAL_TOKEN_SORT_RATIO:
                    if (
                        fuzzy_match(
                            s,
                            transcript,
                            strategy=MatchStrategy.PARTIAL_TOKEN_SORT_RATIO,
                        )
                        >= thresh
                    ):
                        return True

                case MatchRule.PARTIAL_TOKEN_SET_RATIO:
                    if (
                        fuzzy_match(
                            s,
                            transcript,
                            strategy=MatchStrategy.PARTIAL_TOKEN_SET_RATIO,
                        )
                        >= thresh
                    ):
                        return True

                case MatchRule.CONTAINS:
                    if s in transcript:
                        return True

                case MatchRule.EQUALS:
                    if s == transcript:
                        return True

                case MatchRule.STARTS:
                    if transcript.startswith(s):
                        return True

                case MatchRule.ENDS:
                    if transcript.endswith(s):
                        return True

        return False
