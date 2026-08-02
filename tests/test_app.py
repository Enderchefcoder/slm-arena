import ast
import unittest
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "app.py"


class SupraStudioSourceTests(unittest.TestCase):
    """Dependency-free guardrails for the single SupraLabs ZeroGPU Space."""

    def setUp(self):
        self.source = SOURCE.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)

    def test_four_official_supralabs_checkpoints_are_registered(self):
        namespace = {}
        assignments = [
            node for node in self.tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id.endswith("_MODEL") for target in node.targets)
        ]
        exec(compile(ast.Module(body=assignments, type_ignores=[]), str(SOURCE), "exec"), namespace)
        self.assertEqual(
            {namespace["INSTRUCT_MODEL"], namespace["NTP_MODEL"], namespace["TITLE_MODEL"], namespace["THINKING_SUMMARIZER_MODEL"]},
            {
                "SupraLabs/Supra-1.5-50M-Instruct-exp",
                "SupraLabs/Supra-1.5-50M-Base-exp",
                "SupraLabs/supra-title-50m-pre",
                "SupraLabs/reasoning-summarizer-800m-pre",
            },
        )

    def test_each_demo_is_zero_gpu_decorated(self):
        for handler in ("run_instruct", "run_ntp", "run_title", "run_thinking_summarizer"):
            node = next(item for item in self.tree.body if isinstance(item, ast.FunctionDef) and item.name == handler)
            self.assertTrue(node.decorator_list, handler)
            self.assertIn("GPU", ast.unparse(node.decorator_list[0]))

    def test_specialized_prompt_contracts_are_preserved(self):
        self.assertIn('f"User: {message}\\nTitle: "', self.source)
        self.assertIn('reasoning + "\\n"', self.source)
        self.assertIn("apply_chat_template", self.source)
        self.assertIn("json.loads(raw)", self.source)


if __name__ == "__main__":
    unittest.main()
