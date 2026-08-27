import unittest
from types import SimpleNamespace

from gui import ISSUE_KEY_MAP, RamanAnnotationApp


class DummyVariable:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class KeyboardQCTests(unittest.TestCase):
    def setUp(self):
        # 不创建真实窗口，只验证键盘输入与QC变量之间的映射。
        self.app = RamanAnnotationApp.__new__(RamanAnnotationApp)
        self.app.status_var = DummyVariable("")
        self.app.issue_vars = {
            field: DummyVariable(False) for field in ISSUE_KEY_MAP.values()
        }
        self.app._focus_is_text_input = lambda: False

    def press(self, char):
        return self.app._qc_keyboard_input(SimpleNamespace(char=char))

    def test_digits_2_to_8_toggle_each_issue(self):
        for key, field in ISSUE_KEY_MAP.items():
            self.assertEqual(self.press(key), "break")
            self.assertTrue(self.app.issue_vars[field].get())
            self.assertEqual(self.press(key), "break")
            self.assertFalse(self.app.issue_vars[field].get())

    def test_digit_1_is_unused(self):
        self.assertIsNone(self.press("1"))
        self.assertFalse(any(variable.get() for variable in self.app.issue_vars.values()))

    def test_p_sets_pass_and_clears_issues(self):
        self.app.issue_vars["low_snr"].set(True)
        self.assertEqual(self.press("P"), "break")
        self.assertEqual(self.app.status_var.get(), "pass")
        self.assertFalse(any(variable.get() for variable in self.app.issue_vars.values()))

    def test_f_sets_fail_without_changing_issues(self):
        self.app.issue_vars["cosmic_rays"].set(True)
        self.assertEqual(self.press("f"), "break")
        self.assertEqual(self.app.status_var.get(), "fail")
        self.assertTrue(self.app.issue_vars["cosmic_rays"].get())

    def test_text_input_focus_does_not_intercept_keys(self):
        self.app._focus_is_text_input = lambda: True
        self.assertIsNone(self.press("5"))
        self.assertIsNone(self.press("p"))
        self.assertEqual(self.app.status_var.get(), "")
        self.assertFalse(any(variable.get() for variable in self.app.issue_vars.values()))


if __name__ == "__main__":
    unittest.main()
