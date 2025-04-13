#!/bin/bash
rm -f error.txt

if [ -f "${1}" ] ; then
	docker run --rm \
				-v `pwd`:/data \
				flair-brainage-v1.0 \
				compute_brainAge.sh \
				/data/${1} ${2}
else
	docker run --rm \
				-it \
				-v `pwd`:/data \
				flair-brainage-v1.0 \
				bash
fi
