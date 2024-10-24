import subprocess
import os

os.chdir('rust_optimized')
subprocess.run(["maturin", "develop", "--release"])
