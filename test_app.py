import sys
import os
import unittest
from fastapi.testclient import TestClient
from main import app, DB_PATH, init_db
from io import BytesIO
from PIL import Image

class TestVisionIQApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        init_db()

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("VisionIQ", response.text)

    def test_get_analyze_redirect(self):
        response = self.client.get("/analyze", follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertIn("/#describer", response.headers["location"])

    def test_invalid_file_upload(self):
        res = self.client.post(
            "/analyze",
            files={"file": ("test.txt", b"Hello world text", "text/plain")}
        )
        self.assertEqual(res.status_code, 200)
        self.assertIn("Uploaded file is not an image", res.text)

    def test_dummy_image_upload_and_export(self):
        # Create a small 100x100 RGB image in memory
        img = Image.new('RGB', (100, 100), color='red')
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        res = self.client.post(
            "/analyze",
            files={"file": ("sample.png", img_bytes, "image/png")}
        )
        self.assertEqual(res.status_code, 200)

    def test_delete_history(self):
        res = self.client.post("/delete-history/99999")
        self.assertEqual(res.status_code, 200)

    def test_clear_history(self):
        res = self.client.post("/clear-history")
        self.assertEqual(res.status_code, 200)

if __name__ == "__main__":
    unittest.main()
