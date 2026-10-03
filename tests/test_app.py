import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

class AppTests(unittest.TestCase):
    def test_demo_flow(self):
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)
        self.assertEqual(len(app.exception),0)
        app.button[0].click().run()
        self.assertEqual(len(app.exception),0)
        confirm=next(c for c in app.checkbox if c.label.startswith('I have checked'))
        confirm.check().run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(len(app.metric),2)
        app.slider[0].set_value(500).run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(app.metric[1].value,'Inside')

if __name__ == '__main__': unittest.main()
