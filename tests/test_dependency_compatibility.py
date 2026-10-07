"""A tiny offline CPU check of the unchanged demo architecture and checkpoint format."""
import io
import importlib.util
from pathlib import Path
import unittest

import torch

torch.set_num_threads(1)
module_path = Path(__file__).resolve().parents[1] / "demo" / "models.py"
spec = importlib.util.spec_from_file_location("demo_models", module_path)
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)


class DependencyCompatibilityTests(unittest.TestCase):
    def test_lstm_state_dict_and_forward_are_preserved(self):
        model = models.LSTMModel(vocab_size=8, mid_token=2).eval()
        inputs = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
        with torch.no_grad():
            expected = model(inputs)
        self.assertEqual(expected.shape, (2, 2))
        self.assertEqual(model.lstm.hidden_size, 32)
        self.assertEqual(model.lstm.num_layers, 2)
        self.assertTrue(model.lstm.bidirectional)
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = models.LSTMModel(vocab_size=8, mid_token=2).eval()
        restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        with torch.no_grad():
            torch.testing.assert_close(restored(inputs), expected)


if __name__ == "__main__":
    unittest.main()
