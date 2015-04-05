import sys, os, os.path
from mango.application.main_driver import *

if (__name__ == "__main__"):
    argParser = getArgumentParser()
    if haveArgParse:
        args = argParser.parse_args()
        argv = []
    else:
        (args, argv) = argParser.parse_args()

    exeName = sys.executable
    run_main_driver(args.base, args.inputDirectory, exeName)
