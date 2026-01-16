# import from standard library packages
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# create class to hold session information from https://adventofcode.com/
@dataclass
class AoCSession:
    session_id: str
    default_fle_path: Path = Path('config/session_id.txt')

    @classmethod
    def from_file(cls, file_path: Optional[Path | str] = ''):
        """
        returns an instance of cls, with session_id read either from the file at
        the provided filepath or from cls.default_fle_path

        to find a session id, log into https://adventofcode.com/ using a web
        browser, and use developer tools to copy the value of a cookie named
        "session" associated with the domain adventofcode.com

        as of this writing, session_id is a 128-digit hexadecimal number
        """
        if not file_path:
            file_path = cls.default_fle_path
        session_file_path = Path(file_path)
        session_id = session_file_path.read_text().strip()
        return cls(session_id)


# inline test code follows
if __name__ == '__main__':

    import os
    os.chdir('../')

    print(AoCSession.from_file().session_id)
