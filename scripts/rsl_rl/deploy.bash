#!/bin/bash -i

CMD="deploy_xbot2.py --task Isaac-Velocity-Flat-KyonSpotLike-PLAY-v0 --interactive"

python $CMD --checkpoint $1
