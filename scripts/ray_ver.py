import pkg_resources
import os
import time

dists = [d for d in pkg_resources.working_set]
for package in dists:
    if "ray" in repr(package):
        print "%s: %s" % (package, time.ctime(os.path.getctime(package.location)))
