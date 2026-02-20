import time

import ray

from setu.ray import SetuCluster

# connect to ray cluster
ray.init()

# setup setu cluster
setu_cluster = SetuCluster()
info = setu_cluster.start()

print(info)

while True:
    time.sleep(1)
