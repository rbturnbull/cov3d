from fastapp.testing import FastAppTestCase
from cov3d.apps import Cov3d


class TestCov3d(FastAppTestCase):
    app_class = Cov3d

    def test_model_shape(self):
        pass