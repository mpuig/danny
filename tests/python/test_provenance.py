import tempfile
import unittest
from pathlib import Path

from jev.data import sha256_file
from jev.provenance import model_identity, model_path


class ProvenanceTests(unittest.TestCase):
    def test_local_model_identity_hashes_assets_not_unrelated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "config.json").write_text('{"model_type":"fixture"}')
            (path / "model.safetensors").write_bytes(b"fixture weights")
            (path / ".env").write_text("DO_NOT_CAPTURE=secret")
            identity = model_identity(str(path))
            self.assertEqual(model_path(str(path)), path.resolve())
            self.assertIsNone(identity["snapshot_revision"])
            self.assertNotIn(".env", identity["files"])
            self.assertEqual(identity["files"]["model.safetensors"], sha256_file(path / "model.safetensors"))


if __name__ == "__main__":
    unittest.main()
