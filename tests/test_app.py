import ast
import unittest
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "app.py"


class ArenaSourceTests(unittest.TestCase):
    """Dependency-free guardrails for Space configuration and model coverage."""

    def setUp(self):
        self.source = SOURCE.read_text()
        self.tree = ast.parse(self.source)

    def test_top_twenty_registry_is_present(self):
        namespace = {}
        # Execute only constant assignments, never app/UI initialization.
        constants = [node for node in self.tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "MODELS" for t in node.targets)]
        exec(compile(ast.Module(body=constants, type_ignores=[]), str(SOURCE), "exec"), namespace)
        models = namespace["MODELS"]
        self.assertEqual(len(models), 20)
        self.assertEqual(len({model for model, _ in models}), 20)
        self.assertTrue(all("/" in model for model, _ in models))

    def test_zero_gpu_and_immutable_vote_storage_are_configured(self):
        self.assertIn("@GPU(duration=120)", self.source)
        self.assertIn('VOTE_REPO = "Enderchef/slm-arena-votes"', self.source)
        self.assertIn('path_in_repo=f"votes/{record[\'id\']}.json"', self.source)

    def test_reply_uses_chat_templates(self):
        self.assertIn("apply_chat_template", self.source)
        self.assertIn('mode == "Reply"', self.source)


if __name__ == "__main__":
    unittest.main()
