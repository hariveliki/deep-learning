import unittest
import utils


class TestUtils(unittest.TestCase):
    def test_get_dim_after_conv_1(self):
        dim = 64
        kernel_size = 3
        actual = utils.get_dim_after_conv(dim, kernel_size)
        self.assertEqual(actual, 62)


    def test_get_dim_after_conv_2(self):
        dim = 64
        kernel_size = 3
        stride = 2
        actual = utils.get_dim_after_conv(dim, kernel_size, stride)
        self.assertEqual(actual, 31)


    def test_get_dim_after_conv_3(self):
        dim = 64
        kernel_size = 3
        stride = 2
        padding = 1
        actual = utils.get_dim_after_conv(dim, kernel_size, stride, padding)
        self.assertEqual(actual, 32)


    def test_get_dim_after_pool_1(self):
        dim = 64
        kernel_size = 2
        actual = utils.get_dim_after_pool(dim, kernel_size)
        self.assertEqual(actual, 32)


    def test_get_dim_after_pool_2(self):
        dim = 64
        kernel_size = 2
        stride = 2
        actual = utils.get_dim_after_pool(dim, kernel_size, stride)
        self.assertEqual(actual, 32)


    def test_get_dim_after_pool_3(self):
        dim = 64
        kernel_size = 2
        stride = 2
        padding = 1
        actual = utils.get_dim_after_pool(dim, kernel_size, stride, padding)
        self.assertEqual(actual, 33)


    def test_get_dim_after_pool_4(self):
        dim = 64
        kernel_size = 2
        stride = None
        padding = 1
        actual = utils.get_dim_after_pool(dim, kernel_size, stride, padding)
        self.assertEqual(actual, 33)

    
    def test_get_dim_after_conv_pool_1(self):
        dim = utils.get_dim_after_conv(
            dim=64,
            conv_ksize=3,
            conv_stride=1,
            conv_padding=0
        )
        actual = utils.get_dim_after_pool(
            dim=dim,
            pool_kernel_size=2
        )
        self.assertEqual(actual, 31)


    def test_get_dim_after_conv_pool_2(self):
        dim = utils.get_dim_after_conv(
            dim=64,
            conv_ksize=3,
            conv_stride=1,
            conv_padding=0
        )
        dim = utils.get_dim_after_pool(
            dim=dim,
            pool_kernel_size=2
        )
        dim = utils.get_dim_after_conv(
            dim=dim,
            conv_ksize=3,
            conv_stride=2,
            conv_padding=0
        )
        actual = utils.get_dim_after_pool(
            dim=dim,
            pool_kernel_size=2
        )
        self.assertEqual(actual, 7)


    def test_get_dim_after_conv_pool_3(self):
        dim = utils.get_dim_after_conv(
            dim=64,
            conv_ksize=3,
            conv_stride=1,
            conv_padding=0
        )
        dim = utils.get_dim_after_pool(
            dim=dim,
            pool_kernel_size=2
        )
        dim = utils.get_dim_after_conv(
            dim=dim,
            conv_ksize=3,
            conv_stride=2,
            conv_padding=0
        )
        dim = utils.get_dim_after_pool(
            dim=dim,
            pool_kernel_size=2
        )
        dim = utils.get_dim_after_conv(
            dim=dim,
            conv_ksize=3,
            conv_stride=1,
            conv_padding=0
        )
        actual = utils.get_dim_after_pool(
            dim=dim,
            pool_kernel_size=2
        )
        self.assertEqual(actual, 2)


    def test_get_dim_after_conv_pool_5(self):
        dim = utils.get_dim_after_conv(
            dim=64,
            conv_ksize=3,
            conv_stride=1,
            conv_padding=0
        )
        dim = utils.get_dim_after_pool(
            dim=dim,
            pool_kernel_size=2
        )
        dim = utils.get_dim_after_conv(
            dim=dim,
            conv_ksize=3,
            conv_stride=2,
            conv_padding=0
        )
        # dim = utils.get_dim_after_pool(
        #     dim=dim,
        #     pool_kernel_size=2
        # )
        dim = utils.get_dim_after_conv(
            dim=dim,
            conv_ksize=3,
            conv_stride=1,
            conv_padding=0
        )
        dim = utils.get_dim_after_pool(
            dim=dim,
            pool_kernel_size=2
        )
        actual = utils.get_dim_after_conv(
            dim=dim,
            conv_ksize=3,
            conv_stride=1,
            conv_padding=0
        )
        self.assertEqual(actual, 4)


    def test_get_dim_after_conv_and_pool_1(self):
        dim_init = 64
        layers = ["C", "P"]
        confs = [
            {"kernel": 3, "stride": 1, "padding": 0},
            {"kernel": 2}]
        actual = utils.get_dim_after_conv_and_pool(
            dim_init,
            layers,
            confs
        )
        self.assertEqual(actual, 31)


    def test_get_dim_after_conv_and_pool_2(self):
        dim_init = 64
        layers = ["C", "P", "C"]
        confs = [
            {"kernel": 3, "stride": 1, "padding": 0},
            {"kernel": 2},
            {"kernel": 3, "stride": 2, "padding": 0}]
        actual = utils.get_dim_after_conv_and_pool(
            dim_init,
            layers,
            confs
        )
        self.assertEqual(actual, 15)


    def test_get_dim_after_conv_and_pool_3(self):
        dim_init = 64
        layers = ["C", "P", "C", "P"]
        confs = [
            {"kernel": 3, "stride": 1, "padding": 0},
            {"kernel": 2},
            {"kernel": 3, "stride": 2, "padding": 0},
            {"kernel": 2}]
        actual = utils.get_dim_after_conv_and_pool(
            dim_init,
            layers,
            confs
        )
        self.assertEqual(actual, 7)


    def test_get_dim_after_conv_and_pool_4(self):
        dim_init = 64
        layers = ["C", "P", "C", "P", "C"]
        confs = [
            {"kernel": 3, "stride": 1, "padding": 0},
            {"kernel": 2},
            {"kernel": 3, "stride": 2, "padding": 0},
            {"kernel": 2},
            {"kernel": 3, "stride": 1, "padding": 0}]
        actual = utils.get_dim_after_conv_and_pool(
            dim_init,
            layers,
            confs
        )
        self.assertEqual(actual, 5)


    def test_get_dim_after_conv_and_pool_5(self):
        dim_init = 64
        layers = ["C", "P", "C", "P", "C", "P"]
        confs = [
            {"kernel": 3, "stride": 1, "padding": 0},
            {"kernel": 2},
            {"kernel": 3, "stride": 2, "padding": 0},
            {"kernel": 2},
            {"kernel": 3, "stride": 1, "padding": 0},
            {"kernel": 2}]
        actual = utils.get_dim_after_conv_and_pool(
            dim_init,
            layers,
            confs
        )
        self.assertEqual(actual, 2)


    def test_get_dim_after_conv_and_pool_6(self):
        dim_init = 64
        layers = ["C", "P", "C", "P", "C", "C"]
        confs = [
            {"kernel": 3, "stride": 1, "padding": 0},
            {"kernel": 2},
            {"kernel": 3, "stride": 2, "padding": 0},
            {"kernel": 2},
            {"kernel": 3, "stride": 1, "padding": 0},
            {"kernel": 3, "stride": 2, "padding": 0}]
        actual = utils.get_dim_after_conv_and_pool(
            dim_init,
            layers,
            confs
        )
        self.assertEqual(actual, 2)


    def test_get_dim_after_conv_and_pool_7(self):
        dim_init = 64
        layers = ["C", "P", "C", "P", "C", "P", "C"]
        confs = [
            {"kernel": 3, "stride": 1, "padding": 0},
            {"kernel": 2},
            {"kernel": 3, "stride": 2, "padding": 0},
            {"kernel": 2},
            {"kernel": 3, "stride": 1, "padding": 0},
            {"kernel": 2},
            {"kernel": 3, "stride": 1, "padding": 0}]
        actual = utils.get_dim_after_conv_and_pool(
            dim_init,
            layers,
            confs
        )
        self.assertEqual(actual, 0)


if __name__ == "__main__":
    unittest.main()
