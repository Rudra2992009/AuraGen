import unittest
import torch
from model.neural_architecture import create_auragen_model

class TestAuraGenModel(unittest.TestCase):
    def setUp(self):
        self.config = {
            'dim': 2048,
            'num_chains': 12,
            'max_frames': 1800,
            'video_resolution': (320, 240)
        }
        self.model = create_auragen_model(self.config)
        self.prompt_tokens = torch.randint(0, 50000, (1,512))
    def test_forward_shapes(self):
        outputs = self.model(self.prompt_tokens, num_frames=120)
        self.assertIn('video_latent', outputs)
        self.assertTrue(outputs['video_latent'].shape[0] == 1)
        self.assertIn('dialogue', outputs)
        self.assertIn('music', outputs)

if __name__ == "__main__":
    unittest.main()
