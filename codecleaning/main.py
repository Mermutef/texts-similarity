import os
import subprocess

if __name__ == '__main__':
    exclude = {'.idea', '.venv'}
    for root, dirnames, filenames in os.walk("../"):
        dirnames[:] = [dir for dir in dirnames if dir not in exclude]
        for filename in filenames:
            if filename.endswith(".py") and filename != "__init__.py":
                subprocess.run(
                    ["autopep8", "--in-place", "--aggressive", f"{os.path.join(root, filename)}"])
