# import from standard library packages
from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple


# create class to test solutions to puzzles from https://adventofcode.com/
@dataclass
class AoCTester:
    test_cases: List[Tuple[Any, int]] = field(default_factory=list)

    def add_test_case(self, data: Any, output: int) -> None:
        """ add test case as the tuple (data, output) """
        self.test_cases.append((data, output))

    def clear_test_case(self) -> None:
        """ removes all test case """
        self.test_cases = []

    def run_tests(self, test_function: Callable) -> None:
        """ output the results of running all test cases """
        if not self.test_cases:
            print("WARNING! There are no test cases to run.\n")
        for test_case in self.test_cases:
            data = test_case[0]
            expected_output = test_case[1]
            actual_output = test_function(data)
            outputs_equal = expected_output == actual_output
            print(f'{data[:5]=}')
            print(f'{expected_output=}')
            print(f'{actual_output=}')
            print(f'{outputs_equal=}\n')
            assert outputs_equal


# inline test code follows
if __name__ == '__main__':

    test_input = list(range(11))
    test_output = 55
    test_fcn = sum
    tester = AoCTester()

    tester.add_test_case(test_input, test_output)
    tester.run_tests(test_fcn)
    tester.clear_test_case()
    tester.run_tests(test_fcn)
