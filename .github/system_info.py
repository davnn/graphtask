import json
import multiprocessing
import platform
import sys

import graphtask


def get_system_info():
    info = {
        "library": graphtask.version,
        "python": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cores": multiprocessing.cpu_count(),
    }
    return info


if __name__ == "__main__":
    print(json.dumps(get_system_info(), indent=4))
