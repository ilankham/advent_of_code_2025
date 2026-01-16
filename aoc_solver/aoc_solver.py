# import from standard library packages
from dataclasses import dataclass, field
from pathlib import Path
import re
from string import Template
from typing import Any

# import from third-party packages
import requests

# import from local packages
if __name__ == '__main__':
    from aoc_data import AoCData
    from aoc_session import AoCSession
else:
    from .aoc_data import AoCData
    from .aoc_session import AoCSession

# set url and file output filepath parameters
DATA_URL_TEMPLATE = Template(
    'https://adventofcode.com/${year}/day/${day}/input'
)
INSTRUCTIONS_URL_TEMPLATE = Template(
    'https://adventofcode.com/${year}/day/${day}'
)
BASE_OUTPUT_PATH: Path = Path('downloaded_files')

# create class to use for solving puzzles from https://adventofcode.com/
@dataclass
class AoCSolver:
    year: int
    day: int
    session: AoCSession
    data_url: Template = DATA_URL_TEMPLATE
    instructions_url: Template = INSTRUCTIONS_URL_TEMPLATE
    base_output_path: Path = BASE_OUTPUT_PATH
    puzzle_output_path: Path = field(init=False)
    puzzle_data_filepath: Path = field(init=False)
    puzzle_instructions_filepath: Path = field(init=False)

    def __post_init__(self) -> None:
        """ initialize remaining variables and create output filepaths """
        self.puzzle_output_path = (
            self.base_output_path / f'y{self.year}' / f'd{self.day:02}'
        )
        self.puzzle_data_filepath = (
            self.puzzle_output_path / 'data.txt'
        )
        self.puzzle_instructions_filepath = (
            self.puzzle_output_path / 'instructions.html'
        )

    def download_from_aoc_website(
        self,
        url: str,
        filepath: Path,
        overwrite: bool = False,
    ) -> None:
        """
            download a file from the given url and output the results at the
            provided filepath, optionally overwriting any existing file
        """
        output_directory = filepath.parent
        output_directory.mkdir(parents=True, exist_ok=True)
        if overwrite or not filepath.exists():
            print(f'Now downloading {filepath.resolve()}')
            download_response = requests.get(
                url,
                timeout=60,
                cookies={"session": self.session.session_id}
            )
            if download_response.status_code != requests.codes.ok:
                print(f'WARNING! {download_response.status_code=}')
                print(f'{download_response.content=}\n')
            filepath.write_bytes(download_response.content)

    def download_puzzle_input(self, overwrite: bool = False) -> None:
        """
            download user data, optionally overwriting any existing file
        """
        self.download_from_aoc_website(
            url=self.data_url.substitute(year=self.year, day=self.day),
            filepath=self.puzzle_data_filepath,
            overwrite=overwrite,
        )

    def download_instructions(self, overwrite: bool = False) -> None:
        """
            download user instructions, optionally overwriting any existing file
        """
        self.download_from_aoc_website(
            url=self.instructions_url.substitute(year=self.year, day=self.day),
            filepath=self.puzzle_instructions_filepath,
            overwrite=overwrite,
        )

    @property
    def puzzle_input(self):
        """ return user data, downloading the file if needed """
        return_filepath = self.puzzle_data_filepath
        if not return_filepath.exists():
            self.download_puzzle_input()
        return AoCData(return_filepath.read_text())

    @property
    def puzzle_instructions(self) -> str:
        """ return user instructions, downloading the file if needed """
        return_filepath = self.puzzle_instructions_filepath
        if not return_filepath.exists():
            self.download_instructions()
        return return_filepath.read_text()

    def get_value_after(self, prefix: str) -> AoCData:
        """ extract code element after specified text """
        instructions = self.puzzle_instructions
        instructions_re = re.compile(
            fr'{prefix}.*?<code>(<pre>|<em>)?(.*?)(</pre>|</em>)?</code>',
            re.MULTILINE | re.DOTALL | re.IGNORECASE
        )
        instructions_matching_results = instructions_re.search(instructions)
        instructions_matching_text = instructions_matching_results.group(2)
        print(f'{instructions_matching_text=}\n')
        return AoCData(instructions_matching_text)

    def solve_part1(self, data: Any) -> int:
        """ this method should be implemented when subclassing """
        raise NotImplementedError('method solve_part1 is not defined')

    def solve_part2(self, data: Any) -> int:
        """ this method should be implemented when subclassing """
        raise NotImplementedError('method solve_part2 is not defined')


# inline test code follows
if __name__ == '__main__':

    import os
    os.chdir('../')

    # set puzzle parameters and create AoC session
    PUZZLE_DAY = 1
    PUZZLE_YEAR = 2024

    # import from local packages
    from aoc_session import AoCSession
    from aoc_tester import AoCTester
    AoC_SESSION = AoCSession.from_file()

    # import from third-party packages
    import polars
    from polars import DataFrame

    # create solver class and instance
    class Solver(AoCSolver):
        def solve_part1(self, data: DataFrame) -> int:
            return sum((data['col1'].sort() - data['col2'].sort()).abs())

        def solve_part2(self, data: DataFrame) -> int:
            return (
                data
                .join(
                    data['col2'].value_counts(name='col2_counts'),
                    left_on='col1',
                    right_on='col2',
                    how='left'
                )
                .fill_null(0)
                .with_columns(
                    similarity=polars.col('col1')*polars.col('col2_counts')
                )
                .get_column('similarity')
                .sum()
            )

    solver = Solver(PUZZLE_YEAR, PUZZLE_DAY, AoC_SESSION)

    # build part 1 test case
    puzzle_instructions = solver.puzzle_instructions
    part1_test_input = solver.get_value_after('For example:').as_polars
    print(f'{part1_test_input=}\n')
    part1_test_output = solver.get_value_after(', a total distance of ').as_int
    print(f'{part1_test_output=}\n')

    part_1_tester = AoCTester()
    part_1_tester.add_test_case(part1_test_input, part1_test_output)

    part_1_tester.run_tests(solver.solve_part1)

    # determine part 1 solution
    puzzle_input = solver.puzzle_input.as_polars
    part1_solution = solver.solve_part1(puzzle_input)
    print(f'{part1_solution=}\n')

    # add Part 1 Solution to Part 1 Test Cases
    part_1_tester.add_test_case(puzzle_input, part1_solution)

    part_1_tester.run_tests(solver.solve_part1)

    # build part 2 test case
    solver.download_instructions(overwrite=True)

    part2_test_input = part1_test_input
    print(f'{part2_test_input=}\n')
    part2_test_output = solver.get_value_after('end of this process is ').as_int
    print(f'{part2_test_output=}\n')

    part_2_tester = AoCTester()
    part_2_tester.add_test_case(part2_test_input, part2_test_output)

    part_2_tester.run_tests(solver.solve_part2)

    # determine part 1 solution
    part2_solution = solver.solve_part2(puzzle_input)
    print(f'{part2_solution=}\n')
