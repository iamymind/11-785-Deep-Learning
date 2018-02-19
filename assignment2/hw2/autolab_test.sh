#!/bin/bash
docker run -it -v `pwd`:/host cmudeeplearning11785/machine_learning_image \
sh -c "cd /host && ls && python3 ./all_cnn.py"
