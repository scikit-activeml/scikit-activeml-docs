import shutil
import argparse
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--version', dest='version', type=str, default=1)
args = parser.parse_args()
cmd_args = vars(args)
version = cmd_args['version']

version_split = version.split(".")
source_dir = "latest"
version_short_string = f"{version_split[0]}.{version_split[1]}"
if os.path.isdir(version_short_string):
    shutil.rmtree(version_short_string)

shutil.copytree(source_dir, version_short_string)