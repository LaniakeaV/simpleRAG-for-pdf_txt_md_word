import os
import tempfile
import unittest

from rag_backend import RAGSystem


class RAGSystemSmokeTest(unittest.TestCase):
    def test_supported_file_discovery(self):
        rag = RAGSystem.__new__(RAGSystem)
        with tempfile.TemporaryDirectory() as tmp_dir:
            md_path = os.path.join(tmp_dir, "note.md")
            png_path = os.path.join(tmp_dir, "image.png")

            with open(md_path, "w", encoding="utf-8") as file:
                file.write("# hello")
            with open(png_path, "w", encoding="utf-8") as file:
                file.write("not supported")

            files = rag._get_supported_files(tmp_dir)

        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].endswith("note.md"))

    def test_index_fingerprint_is_stable(self):
        rag = RAGSystem.__new__(RAGSystem)
        with tempfile.TemporaryDirectory() as tmp_dir:
            txt_path = os.path.join(tmp_dir, "doc.txt")
            with open(txt_path, "w", encoding="utf-8") as file:
                file.write("hello world")

            files = rag._get_supported_files(tmp_dir)
            first = rag._build_index_fingerprint(tmp_dir, files)
            second = rag._build_index_fingerprint(tmp_dir, files)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)

    def test_get_response_requires_llm_config(self):
        rag = RAGSystem.__new__(RAGSystem)
        rag.retriever = object()

        self.assertEqual(rag.get_response("hello", "", "model", "https://example.com/v1"), "请先配置 API Key。")
        self.assertEqual(rag.get_response("hello", "sk-test", "", "https://example.com/v1"), "请先配置模型名称。")
        self.assertEqual(rag.get_response("hello", "sk-test", "model", ""), "请先配置 API 地址。")


if __name__ == "__main__":
    unittest.main()