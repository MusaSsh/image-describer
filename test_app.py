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

    def test_signup_and_login_flow(self):
        # Signup test user
        test_email = f"test_{os.urandom(4).hex()}@example.com"
        signup_res = self.client.post(
            "/signup",
            data={"full_name": "Test User", "email": test_email, "password": "securepassword123"},
            follow_redirects=False
        )
        self.assertEqual(signup_res.status_code, 303)
        self.assertIn("visioniq_user_id", signup_res.cookies)

        # Logout
        logout_res = self.client.get("/logout", follow_redirects=False)
        self.assertEqual(logout_res.status_code, 303)

        # Login
        login_res = self.client.post(
            "/login",
            data={"email": test_email, "password": "securepassword123"},
            follow_redirects=False
        )
        self.assertEqual(login_res.status_code, 303)
        self.assertIn("visioniq_user_id", login_res.cookies)

    def test_invalid_login(self):
        res = self.client.post(
            "/login",
            data={"email": "nonexistent@example.com", "password": "wrongpassword"},
            follow_redirects=False
        )
        self.assertEqual(res.status_code, 200)
        self.assertIn("Invalid email or password", res.text)

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

if __name__ == "__main__":
    unittest.main()
