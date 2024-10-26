import unittest
import torch
import numpy as np
from torchtmpl.utils import topk_accuracy

class TestTopKAccuracy(unittest.TestCase):
    def test_topk_accuracy(self):
        # Test case 1
        outputs = torch.tensor([[0.5, 0.2, 0.2],
                                [0.3, 0.4, 0.2],
                                [0.2, 0.4, 0.3],
                                [0.7, 0.2, 0.1]])
        targets = torch.tensor([0, 1, 2, 2])
        k = 2
        result = topk_accuracy(outputs, targets, k)
        print(result)
        expected_result = 0.75
        self.assertAlmostEqual(result, expected_result, places=5)

            # Test case 2
        outputs_case2 = torch.tensor([[0.3, 0.4, 0.2],
                                    [0.5, 0.2, 0.2],
                                    [0.1, 0.8, 0.1],
                                    [0.6, 0.2, 0.2]])
        targets_case2 = torch.tensor([2, 1, 0, 0])
        k_case2 = 3
        result_case2 = topk_accuracy(outputs_case2, targets_case2, k_case2)
        expected_result_case2 = 1.0
        self.assertAlmostEqual(result_case2, expected_result_case2, places=5)

        # Test case 3
        outputs_case3 = torch.tensor([[0.1, 0.5, 0.4],
                                    [0.2, 0.1, 0.7],
                                    [0.3, 0.3, 0.4],
                                    [0.8, 0.1, 0.1]])
        targets_case3 = torch.tensor([1, 2, 0, 0])
        k_case3 = 1
        result_case3 = topk_accuracy(outputs_case3, targets_case3, k_case3)
        expected_result_case3 = 0.75
        self.assertAlmostEqual(result_case3, expected_result_case3, places=5)

        # Test case 4
        outputs_case4 = torch.tensor([[0.2, 0.3, 0.5],
                                    [0.4, 0.2, 0.4],
                                    [0.1, 0.1, 0.8],
                                    [0.5, 0.2, 0.3]])
        targets_case4 = torch.tensor([0, 1, 2, 2])
        k_case4 = 2
        result_case4 = topk_accuracy(outputs_case4, targets_case4, k_case4)
        print(result_case4)
        expected_result_case4 = 0.5
        self.assertAlmostEqual(result_case4, expected_result_case4, places=5)

if __name__ == '__main__':
    unittest.main()
