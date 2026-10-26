"""CPU regression tests; run: python -m unittest discover -p 'test_*.py' -v.

Synthetic tensors test equations, layouts and gradient hygiene, not clinical
quality. The separate smoke driver exercises real Nifti data end-to-end.
"""
import ast
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from dataset import HipMRISlices, load_slice, verify_patient_splits
from modules import VQVAE, VectorQuantizer, PixelCNNPrior, DCGenerator, DCDiscriminator
from runtime import finite_loss

torch.set_num_threads(2)


class QuantizationTests(unittest.TestCase):
    def test_nearest_codes_and_channel_layout(self):
        torch.manual_seed(4)
        q = VectorQuantizer(8, 3)
        z = torch.randn(2, 3, 5, 4, requires_grad=True)
        quantized, loss, perplexity, codes = q(z)
        flat = z.permute(0, 2, 3, 1).reshape(-1, 3)
        expected_codes = torch.cdist(flat, q.codebook.weight).argmin(1).view(2, 5, 4)
        self.assertTrue(torch.equal(codes, expected_codes))
        expected = q.codebook(codes).permute(0, 3, 1, 2)
        torch.testing.assert_close(quantized, expected, atol=1e-6, rtol=1e-6)
        finite_loss(loss)
        self.assertTrue(1 <= float(perplexity) <= 8.001)

    def test_straight_through_gradient_and_eval_no_mutation(self):
        q = VectorQuantizer(8, 3)
        z = torch.randn(2, 3, 4, 4, requires_grad=True)
        quantized, loss, _, _ = q(z)
        quantized.sum().backward(retain_graph=True)
        torch.testing.assert_close(z.grad, torch.ones_like(z))
        self.assertIsNone(q.codebook.weight.grad)
        q.zero_grad(); z.grad = None
        loss.backward()
        self.assertGreater(float(q.codebook.weight.grad.abs().sum()), 0)
        q.eval()
        before = q.ema_counts.clone()
        with torch.no_grad():
            q(z)
        torch.testing.assert_close(before, q.ema_counts)

    def test_forward_matches_discrete_decode(self):
        model = VQVAE(width=8, latent_dim=4, num_codes=8).eval()
        x = torch.rand(2, 1, 256, 128)
        with torch.no_grad():
            recon, _, _, codes = model(x)
            decoded = model.decode_from_codes(codes)
        self.assertEqual(tuple(recon.shape), tuple(x.shape))
        torch.testing.assert_close(recon, decoded, atol=1e-6, rtol=1e-5)
        self.assertTrue(bool(((recon >= 0) & (recon <= 1)).all()))

    def test_official_vqvae_backward(self):
        model = VQVAE()
        x = torch.rand(2, 1, 256, 128)
        recon, vq, _, _ = model(x)
        loss = F.mse_loss(recon, x) + vq
        finite_loss(loss)
        loss.backward()
        for component in (model.encoder, model.decoder, model.quantizer):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in component.parameters()))


    def test_eightfold_grid_and_l1_backward(self):
        model = VQVAE(width=8, latent_dim=4, num_codes=8, downsample=8)
        x = torch.rand(2, 1, 256, 128)
        recon, loss, _, codes = model(x)
        self.assertEqual(tuple(codes.shape), (2, 32, 16))
        self.assertEqual(tuple(recon.shape), tuple(x.shape))
        (F.l1_loss(recon, x) + loss).backward()
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(x)[0], model.decode_from_codes(model.encode_to_codes(x)),
                                       atol=1e-6, rtol=1e-5)


class PriorTests(unittest.TestCase):
    def test_first_raster_position_can_learn_a_nonuniform_marginal(self):
        torch.manual_seed(9)
        model = PixelCNNPrior(8, width=8, num_blocks=1, grid_hw=(3, 4))
        codes = torch.randint(8, (2, 3, 4))
        target = torch.full((2,), 3, dtype=torch.long)
        initial = model(codes)[:, :, 0, 0].softmax(1)[:, 3].mean().item()
        opt = torch.optim.Adam(model.parameters(), lr=.03)
        for _ in range(16):
            opt.zero_grad()
            loss = F.cross_entropy(model(codes)[:, :, 0, 0], target)
            loss.backward()
            self.assertGreater(float(model.out.conv.bias.grad.abs().sum()), 0)
            opt.step()
        final = model(codes)[:, :, 0, 0].softmax(1)[:, 3].mean().item()
        self.assertGreater(final, initial + .15)

    def test_causality_current_and_future_codes_cannot_leak(self):
        for kernel in (3, 7):
            torch.manual_seed(3)
            model = PixelCNNPrior(8, width=8, num_blocks=2, kernel_size=kernel).eval()
            a = torch.randint(8, (2, 8, 6))
            b = a.clone()
            b.flatten(1)[:, 3 * 6 + 2:] = torch.randint(8, (2, 8 * 6 - 20))
            with torch.no_grad():
                la, lb = model(a), model(b)
            torch.testing.assert_close(la[:, :, 3, 2], lb[:, :, 3, 2],
                                       atol=1e-6, rtol=1e-6)

    def test_cropped_sampler_logits_match_full_grid(self):
        for kernel in (3, 7):
            model = PixelCNNPrior(8, width=8, num_blocks=1, kernel_size=kernel).eval()
            codes = torch.randint(8, (2, 32, 6))
            with torch.no_grad():
                full = model(codes)
                for i in (0, 3, 15, 31):
                    top = max(0, i - model.row_radius)
                    cropped = model(codes[:, top:i + 1])
                    torch.testing.assert_close(full[:, :, i], cropped[:, :, i - top],
                                               atol=1e-6, rtol=1e-5)

    def test_sampling_callback_and_loss_backward(self):
        model = PixelCNNPrior(8, width=8, num_blocks=1, grid_hw=(3, 4))
        codes = torch.randint(8, (2, 3, 4))
        loss = model.loss(codes)
        finite_loss(loss); loss.backward()
        calls = []
        samples = model.sample(2, torch.device("cpu"), progress=lambda d, t: calls.append((d, t)))
        self.assertEqual(tuple(samples.shape), (2, 3, 4))
        self.assertEqual(calls[-1], (12, 12))
        with self.assertRaises(ValueError):
            model.sample(2, torch.device("cpu"), temperature=0)

    def test_official_prior_backward(self):
        model = PixelCNNPrior()
        loss = model.loss(torch.randint(512, (2, 64, 32)))
        finite_loss(loss); loss.backward()
        self.assertGreater(float(model.embedding.weight.grad.abs().sum()), 0)


class BaselineTests(unittest.TestCase):
    def test_official_gan_shapes_and_both_optimizer_steps(self):
        g, d = DCGenerator(), DCDiscriminator()
        og = torch.optim.Adam(g.parameters(), lr=2e-4)
        od = torch.optim.Adam(d.parameters(), lr=2e-4)
        fake = g(torch.randn(2, 100))
        self.assertEqual(tuple(fake.shape), (2, 1, 256, 128))
        self.assertEqual(tuple(d(fake.detach()).shape), (2,))
        dl = F.binary_cross_entropy_with_logits(d(fake.detach()), torch.zeros(2))
        finite_loss(dl); dl.backward(); od.step()
        og.zero_grad()
        gl = F.binary_cross_entropy_with_logits(d(fake), torch.ones(2))
        finite_loss(gl); gl.backward(); og.step()


class DataTests(unittest.TestCase):
    def test_crop_normalisation_and_early_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            for split, pid in (("train", 1), ("validate", 2), ("test", 3)):
                folder = Path(tmp) / f"keras_slices_{split}"
                folder.mkdir()
                for i in range(2):
                    width = 144 if i == 0 else 128
                    data = np.ones((256, width, 1), np.float32) * 9
                    nib.save(nib.Nifti1Image(data, np.eye(4)),
                             folder / f"case_{pid:05}_week_1_slice_{i:03}.nii.gz")
            self.assertEqual(verify_patient_splits(tmp), {"train": 1, "validate": 1, "test": 1})
            ds = HipMRISlices(tmp, "train", limit=1)
            self.assertEqual(len(ds), 1)
            self.assertEqual(tuple(ds[0].shape), (1, 256, 128))
            self.assertEqual(float(ds[0].max()), 1)
            leak = Path(tmp) / "keras_slices_test" / "case_00001_week_1_slice_000.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros((256, 128), np.float32), np.eye(4)), leak)
            with self.assertRaises(AssertionError):
                verify_patient_splits(tmp)

    def test_reject_unexpected_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros((256, 100), np.float32), np.eye(4)), path)
            with self.assertRaises(ValueError):
                load_slice(path)

    def test_modules_dependencies_are_torch_only(self):
        tree = ast.parse((Path(__file__).parent / "modules.py").read_text())
        roots = {node.module.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom)}
        roots |= {alias.name.split(".")[0] for node in ast.walk(tree)
                  if isinstance(node, ast.Import) for alias in node.names}
        self.assertEqual(roots, {"torch"})


if __name__ == "__main__":
    unittest.main()