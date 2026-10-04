"""Loader lifecycle tests using explicit mocks; no scientific results are mocked."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

import model_loader


class ModelLoaderTests(unittest.TestCase):
    def setUp(self):
        self.cache = patch.object(model_loader, "_model", None)
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.model = Mock()
        self.loader = Mock(return_value=self.model)
        self.cuda_available = Mock(return_value=False)
        modules = {
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=self.cuda_available)),
            "transformer_lens": SimpleNamespace(
                HookedTransformer=SimpleNamespace(from_pretrained=self.loader)
            ),
        }
        self.modules = patch.dict("sys.modules", modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_cpu_load_is_cached_and_eval_is_restored(self):
        self.assertIs(model_loader.get_model(), self.model)
        self.assertIs(model_loader.get_model(), self.model)
        self.loader.assert_called_once_with("gpt2", device="cpu")
        self.assertEqual(self.model.eval.call_count, 2)

    def test_cuda_is_selected_when_available(self):
        self.cuda_available.return_value = True
        model_loader.get_model()
        self.loader.assert_called_once_with("gpt2", device="cuda")
        self.model.eval.assert_called_once_with()

    def test_concurrent_calls_share_one_load(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            models = list(pool.map(lambda _: model_loader.get_model(), range(8)))
        self.assertTrue(all(model is self.model for model in models))
        self.loader.assert_called_once()

    def test_failed_load_can_be_retried(self):
        self.loader.side_effect = [RuntimeError("download failed"), self.model]
        with self.assertRaisesRegex(RuntimeError, "download failed"):
            model_loader.get_model()
        self.assertIsNone(model_loader._model)
        self.assertIs(model_loader.get_model(), self.model)


if __name__ == "__main__":
    unittest.main()
