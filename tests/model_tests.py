import unittest

from src.configs.config_classes import InferenceConfig
from src.inference.models import TextLangPredictor


class TestStringMethods(unittest.TestCase):
    def setUp(self):
        cfg = InferenceConfig()
        self.predictor = TextLangPredictor(cfg)

    def test_russian_long(self):
        text = """
        Мечта может стать поддержкой или источником страданий. Мечта может наполнить человека жизнью или убить его.
         Даже если мечта оставила человека, её частица будет всегда тлеть в глубине его сердца.
          Каждый должен хоть раз представить, что его жизнь - это жизнь божьего мученика, преследующего свою мечту.
        """
        langs, spans = zip(*self.predictor.parse_text(text).items())

        self.assertEqual(len(langs), 1, f"Predicted: {langs}, expected: ru")
        self.assertEqual(langs[0], "ru", f"Predicted: {langs}, expected: ru")

    def test_russian_short(self):
        text = "В споре рождается истина"
        langs, spans = zip(*self.predictor.parse_text(text).items())

        self.assertEqual(len(langs), 1, f"Predicted: {langs}, expected: ru")
        self.assertEqual(langs[0], "ru", f"Predicted: {langs}, expected: ru")

    def test_ukrainian_short(self):
        text = "Як тебе звати"
        langs, spans = zip(*self.predictor.parse_text(text).items())

        self.assertEqual(len(langs), 1, f"Predicted: {langs}, expected: uk")
        self.assertEqual(langs[0], "uk", f"Predicted: {langs}, expected: uk")

    def test_ukrainian_short2(self):
        text = "У суперечці народжується істина"
        langs, spans = zip(*self.predictor.parse_text(text).items())

        self.assertEqual(len(langs), 1, f"Predicted: {langs}, expected: uk")
        self.assertEqual(langs[0], "uk", f"Predicted: {langs}, expected: uk")

    def test_english_short(self):
        text = "Truth is born in a dispute"
        langs, spans = zip(*self.predictor.parse_text(text).items())

        self.assertEqual(len(langs), 1, f"Predicted: {langs}, expected: en")
        self.assertEqual(langs[0], "en", f"Predicted: {langs}, expected: en")

    def test_german_short(self):
        text = "Wahrheit wird in einem Streit geboren"
        langs, spans = zip(*self.predictor.parse_text(text).items())

        self.assertEqual(len(langs), 1, f"Predicted: {langs}, expected: de")
        self.assertEqual(langs[0], "de", f"Predicted: {langs}, expected: de")

    def test_belarusian_short(self):
        text = "У спрэчцы нараджаецца ісціна"
        langs, spans = zip(*self.predictor.parse_text(text).items())

        self.assertEqual(len(langs), 1, f"Predicted: {langs}, expected: be")
        self.assertEqual(langs[0], "be", f"Predicted: {langs}, expected: be")

    def test_kazakh_short(self):
        text = "Ақиқат дауда туады"
        langs, spans = zip(*self.predictor.parse_text(text).items())

        self.assertEqual(len(langs), 1, f"Predicted: {langs}, expected: kk")
        self.assertEqual(langs[0], "kk", f"Predicted: {langs}, expected: kk")

    def test_azerbaijani_short(self):
        text = "Həqiqət mübahisədə doğulur"
        langs, spans = zip(*self.predictor.parse_text(text).items())

        self.assertEqual(len(langs), 1, f"Predicted: {langs}, expected: az")
        self.assertEqual(langs[0], "az", f"Predicted: {langs}, expected: az")

    def test_armenian_short(self):
        text = "Ճշմարտությունը ծնվում է վեճի մեջ"
        langs, spans = zip(*self.predictor.parse_text(text).items())

        self.assertEqual(len(langs), 1, f"Predicted: {langs}, expected: hy")
        self.assertEqual(langs[0], "hy", f"Predicted: {langs}, expected: hy")

    def test_georgian_short(self):
        text = "სიმართლე კამათში იბადება"
        langs, spans = zip(*self.predictor.parse_text(text).items())

        self.assertEqual(len(langs), 1, f"Predicted: {langs}, expected: ka")
        self.assertEqual(langs[0], "ka", f"Predicted: {langs}, expected: ka")

    def test_hebrew_short(self):
        text = "האמת נולדת במחלוקת"
        langs, spans = zip(*self.predictor.parse_text(text).items())

        self.assertEqual(len(langs), 1, f"Predicted: {langs}, expected: he")
        self.assertEqual(langs[0], "he", f"Predicted: {langs}, expected: he")

    def test_multilang1(self):
        text = "Great day Замечательный день Wunderschönen Tag"
        langs, spans = zip(*self.predictor.parse_text(text).items())

        actual_langs = ("en", "ru", "de")
        actual_spans = ("Great day", "Замечательный день", "Wunderschönen Tag")

        self.assertEqual(len(langs), len(actual_langs), f"Predicted: {langs}, expected: {actual_langs}")
        self.assertEqual(langs, actual_langs, f"Predicted: {langs}, expected: {actual_langs}")
        self.assertEqual(spans, actual_spans, f"Predicted: {spans}, expected: {actual_spans}")

    def test_multilang2(self):
        text = "Мечта может стать support or source leiden. Mai träumen insanı həyatla doldurmaq немесе оны өлтіріңіз."
        langs, spans = zip(*self.predictor.parse_text(text).items())

        actual_langs = ("ru", "en", "de", "az", "kk")
        actual_spans = (
            "Мечта может стать",
            "support or source",
            "leiden Mai träumen",
            "insanı həyatla doldurmaq",
            "немесе оны өлтіріңіз",
        )

        self.assertEqual(len(langs), len(actual_langs), f"Predicted: {langs}, expected: {actual_langs}")
        self.assertEqual(langs, actual_langs, f"Predicted: {langs}, expected: {actual_langs}")
        self.assertEqual(spans, actual_spans, f"Predicted: {spans}, expected: {actual_spans}")


if __name__ == "__main__":
    unittest.main()
