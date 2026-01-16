# import from standard library packages
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Any, Dict, List, Optional, Tuple

# import from third-party packages
from polars import DataFrame

class Direction(IntEnum):
    N = 0
    NE = 1
    E = 2
    SE = 3
    S = 4
    SW = 5
    W = 6
    NW = 7


class OutOfBounds(Exception):
    pass


@dataclass
class AoCMap:
    df: DataFrame
    x: int = 0
    y: int = 0
    heading: Direction = Direction.N

    def __repr__(self):
        return str(self.df)

    def __getitem__(self, indices: Tuple[int, int]) -> Any:
        x, y = indices
        if x not in range(self.x_max) or y not in range(self.y_max):
            raise OutOfBounds(f'Position {indices} is out of bounds!')
        return self.df[y, x]

    def clone(self):
        return AoCMap(self.df.clone(), self.x, self.y, self.heading)

    def encloses(self, x: int, y: int) -> bool:
        return x in range(self.x_max) and y in range(self.y_max)

    @property
    def x_max(self) -> int:
        _, x_max = self.df.shape
        return x_max

    @property
    def y_max(self) -> int:
        y_max, _ = self.df.shape
        return y_max

    @property
    def shape(self) -> Tuple[int, int]:
        return self.x_max, self.y_max

    @property
    def position(self) -> Tuple[int, int]:
        return self.x, self.y

    @position.setter
    def position(self, value: Tuple[int, int]) -> None:
        x_new, y_new = value
        if not self.encloses(x_new, y_new):
            raise OutOfBounds(f'Position {value} is out of bounds!')
        self.x, self.y = x_new, y_new

    @property
    def at_edge(self) -> bool:
        return (self.x == 0 or self.x == self.x_max - 1) or (
                    self.y == 0 or self.y == self.y_max - 1)

    @property
    def value_counts(self) -> Dict[Any, int]:
        return dict(
            sorted(
                self.df.unpivot().get_column('value').value_counts().iter_rows()
            )
        )

    @property
    def element_positions(self) -> Dict[Any, List[Tuple[int, int]]]:
        return_value = defaultdict(list)
        for x in range(self.x_max):
            for y in range(self.y_max):
                return_value[self.df[y,x]].append((x, y))
        return dict(sorted(return_value.items()))

    def look(self, steps: Optional[int] = None,
             direction: Optional[Direction] = None) -> Optional[List] | Any:
        if steps is None:
            steps = math.inf
        if direction is None:
            direction = self.heading
        if steps == 0:
            return self.df[self.y, self.x]
        if steps < 0:
            direction = Direction((direction + 4) % 8)
            steps = abs(steps)
        max_steps_N = min(steps, self.y)
        max_steps_E = min(steps, (self.x_max - 1) - self.x)
        max_steps_S = min(steps, (self.y_max - 1) - self.y)
        max_steps_W = min(steps, self.x)
        match direction:
            case Direction.N:
                max_steps = range(1, max_steps_N + 1)
                return_value = [self.df[self.y - i, self.x] for i in max_steps]
            case Direction.NE:
                max_steps = range(1, min(max_steps_N, max_steps_E) + 1)
                return_value = [self.df[self.y - i, self.x + i] for i in
                                max_steps]
            case Direction.E:
                max_steps = range(1, max_steps_E + 1)
                return_value = [self.df[self.y, self.x + i] for i in max_steps]
            case Direction.SE:
                max_steps = range(1, min(max_steps_S, max_steps_E) + 1)
                return_value = [self.df[self.y + i, self.x + i] for i in
                                max_steps]
            case Direction.S:
                max_steps = range(1, max_steps_S + 1)
                return_value = [self.df[self.y + i, self.x] for i in max_steps]
            case Direction.SW:
                max_steps = range(1, min(max_steps_S, max_steps_W) + 1)
                return_value = [self.df[self.y + i, self.x - i] for i in
                                max_steps]
            case Direction.W:
                max_steps = range(1, max_steps_W + 1)
                return_value = [self.df[self.y, self.x - i] for i in max_steps]
            case Direction.NW:
                max_steps = range(1, min(max_steps_N, max_steps_W) + 1)
                return_value = [self.df[self.y - i, self.x - i] for i in
                                max_steps]
            case _:
                return_value = None
        return return_value

    def update(self, values: List[Any], offset: int = 0,
               direction: Optional[Direction] = None) -> None:
        steps = len(values)
        if direction is None:
            direction = self.heading
        if offset < 0:
            direction = Direction((direction + 4) % 8)
            offset = abs(offset)
        max_steps_N = min(steps, self.y + 1 - offset)
        max_steps_E = min(steps, self.x_max - self.x - offset)
        max_steps_S = min(steps, self.y_max - self.y - offset)
        max_steps_W = min(steps, self.x + 1 - offset)
        match direction:
            case Direction.N:
                max_steps = range(max_steps_N)
                for i in max_steps:
                    self.df[self.y - i - offset, self.x] = values[i]
            case Direction.NE:
                max_steps = range(min(max_steps_N, max_steps_E))
                for i in max_steps:
                    self.df[self.y - i - offset, self.x + i + offset] = values[
                        i]
            case Direction.E:
                max_steps = range(max_steps_E)
                for i in max_steps:
                    self.df[self.y, self.x + i + offset] = values[i]
            case Direction.SE:
                max_steps = range(min(max_steps_S, max_steps_E))
                for i in max_steps:
                    self.df[self.y + i + offset, self.x + i + offset] = values[
                        i]
            case Direction.S:
                max_steps = range(max_steps_S)
                for i in max_steps:
                    self.df[self.y + i + offset, self.x] = values[i]
            case Direction.SW:
                max_steps = range(min(max_steps_S, max_steps_W))
                for i in max_steps:
                    self.df[self.y + i + offset, self.x - i - offset] = values[
                        i]
            case Direction.W:
                max_steps = range(max_steps_W)
                for i in max_steps:
                    self.df[self.y, self.x - i - offset] = values[i]
            case Direction.NW:
                max_steps = range(min(max_steps_N, max_steps_W))
                for i in max_steps:
                    self.df[self.y - i - offset, self.x - i - offset] = values[
                        i]

    def walk(self, steps: int, direction: Optional[Direction] = None) -> None:
        if direction is None:
            direction = self.heading
        path_length = min(steps, len(self.look(steps, direction)))
        match direction:
            case Direction.N:
                self.y -= path_length  # north
            case Direction.NE:
                self.x += path_length  # east
                self.y -= path_length  # north
            case Direction.E:
                self.x += path_length  # east
            case Direction.SE:
                self.x += path_length  # east
                self.y += path_length  # south
            case Direction.S:
                self.y += path_length  # south
            case Direction.SW:
                self.x -= path_length  # west
                self.y += path_length  # south
            case Direction.W:
                self.x -= path_length  # west
            case Direction.NW:
                self.x -= path_length  # west
                self.y -= path_length  # north

    def rotate(self, degrees: int) -> None:
        sign_adjustment = 0.5 if degrees > 0 else -0.5
        new_heading = (self.heading + int(degrees // 45 + sign_adjustment)) % 8
        self.heading = Direction(new_heading)


if __name__ == "__main__":
    m = AoCMap(
        DataFrame(
            {
                f'col{i}': list(range(10 * i, 10 * (i + 1))) for i in
                range(0, 8)
            }
        )
    )
    print(f'{m=}')
    print(f'{m.heading=}')
    print(f'{m.position=} vs (0,0)')
    print(f'{m[0,0]=}')
    print(f'{m.look(0)=}')
    print(f'{m.at_edge=}\n')

    print(f'{m.look(direction=Direction.E)=}')
    print(f'{m.look(2, Direction.E)=}')
    m.walk(2, Direction.E)
    print(f'{m.position=} vs (2,0)')
    print(f'{m[2,0]=}')
    print(f'{m.look(0)=}')
    print(f'{m.at_edge=}\n')

    print(f'{m.look(direction=Direction.SE)=}')
    print(f'{m.look(4, Direction.SE)=}')
    m.walk(4, Direction.SE)
    print(f'{m.position=} vs (6,4)')
    print(f'{m[6,4]=}')
    print(f'{m.look(0)=}')
    print(f'{m.at_edge=}\n')

    print(f'{m.look(direction=Direction.S)=}')
    print(f'{m.look(2, Direction.S)=}')
    m.walk(2, Direction.S)
    print(f'{m.position=} vs (6,6)')
    print(f'{m[6,6]=}')
    print(f'{m.look(0)=}')
    print(f'{m.at_edge=}\n')

    print(f'{m.look(direction=Direction.SW)=}')
    print(f'{m.look(2, Direction.SW)=}')
    m.walk(2, Direction.SW)
    print(f'{m.position=} vs (4,8)')
    print(f'{m[4,8]=}')
    print(f'{m.look(0)=}')
    print(f'{m.at_edge=}\n')

    print(f'{m.look(direction=Direction.W)=}')
    print(f'{m.look(2, Direction.W)=}')
    m.walk(2, Direction.W)
    print(f'{m.position=} vs (2,8)')
    print(f'{m[2,8]=}')
    print(f'{m.look(0)=}')
    print(f'{m.at_edge=}\n')

    print(f'{m.look(direction=Direction.NW)=}')
    print(f'{m.look(2, Direction.NW)=}')
    m.walk(2, Direction.NW)
    print(f'{m.position=} vs (0,6)')
    print(f'{m[0,6]=}')
    print(f'{m.look(0)=}')
    print(f'{m.at_edge=}\n')

    m.position = (7, 0)
    print(f'{m.position=} vs (7,0)')
    print(f'{m[7,0]=}')
    print(f'{m.look(0)=}')
    print(f'{m.at_edge=}\n')

    m.position = (7, 9)
    print(f'{m.position=} vs (7,9)')
    print(f'{m[7,9]=}')
    print(f'{m.look(0)=}')
    print(f'{m.at_edge=}\n')

    m.position = (0, 9)
    print(f'{m.position=} vs (0,9)')
    print(f'{m[0,9]=}')
    print(f'{m.look(0)=}')
    print(f'{m.at_edge=}\n')

    print(f'{m.heading=}')
    m.rotate(45)
    print(f'{m.heading=} vs NE after rotating 45°')
    m.rotate(90)
    print(f'{m.heading=} vs SE after rotating 90°')
    m.rotate(135)
    print(f'{m.heading=} vs W after rotating 135°')
    m.rotate(180)
    print(f'{m.heading=} vs E after rotating 180°')
    m.rotate(225)
    print(f'{m.heading=} vs NW after rotating 225°')
    m.rotate(270)
    print(f'{m.heading=} vs SW after rotating 270°')
    m.rotate(315)
    print(f'{m.heading=} vs S after rotating 315°')
    m.rotate(360)
    print(f'{m.heading=} vs S after rotating 360°\n')

    m.rotate(-45)
    print(f'{m.heading=} vs SE after rotating -45°')
    m.rotate(-90)
    print(f'{m.heading=} vs NE after rotating -90°')
    m.rotate(-135)
    print(f'{m.heading=} vs W after rotating -135°')
    m.rotate(-180)
    print(f'{m.heading=} vs E after rotating -180°')
    m.rotate(-225)
    print(f'{m.heading=} vs SW after rotating -225°')
    m.rotate(-270)
    print(f'{m.heading=} vs NW after rotating -270°')
    m.rotate(-315)
    print(f'{m.heading=} vs N after rotating -315°')
    m.rotate(-360)
    print(f'{m.heading=} vs N after rotating -360°')

    m.position = (0, 9)
    print(f'{m.position=} vs (0,9)')
    print(f'{m.look(0)=}')
    m.update(values=list(range(90, 100)), direction=Direction.N)
    print(f'{m=}')
    m.update(values=list(range(90, 100)), direction=Direction.N, offset=3)
    print(f'{m=}')
    m.update(values=list(range(90, 100)), direction=Direction.NE)
    print(f'{m=}')
    m.update(values=list(range(90, 100)), offset=3, direction=Direction.NE)
    print(f'{m=}')
    m.update(values=list(range(90, 100)), direction=Direction.E)
    print(f'{m=}')
    m.update(values=list(range(90, 100)), offset=3, direction=Direction.E)
    print(f'{m=}\n')

    m.position = (3, 0)
    print(f'{m.position=} vs (3,0)')
    print(f'{m.look(0)=}')
    m.update(values=list(range(90, 100)), direction=Direction.SE)
    print(f'{m=}')
    m.update(values=list(range(90, 100)), direction=Direction.SE, offset=3)
    print(f'{m=}')
    m.update(values=list(range(90, 100)), direction=Direction.S)
    print(f'{m=}')
    m.update(values=list(range(90, 100)), direction=Direction.S, offset=3)
    print(f'{m=}')
    m.update(values=list(range(90, 100)), direction=Direction.SW)
    print(f'{m=}')
    m.update(values=list(range(90, 100)), direction=Direction.SW, offset=3)
    print(f'{m=}')
    m.update(values=list(range(90, 100)), direction=Direction.W)
    print(f'{m=}')
    m.update(values=list(range(90, 100)), direction=Direction.W, offset=3)
    print(f'{m=}\n')

    m.position = (7, 9)
    print(f'{m.position=} vs (7,9)')
    print(f'{m.look(0)=}')
    m.update(values=list(range(90, 100)), direction=Direction.NW)
    print(f'{m=}')
    m.update(values=list(range(90, 100)), direction=Direction.NW, offset=3)
    print(f'{m=}\n')

    print(f'{m.value_counts=}\n')

    print(f'{m.element_positions=}\n')
