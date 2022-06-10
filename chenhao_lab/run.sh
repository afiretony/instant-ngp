#!usr/bin/bash
python chenhao_lab/run.py --mode="nerf" \
                --scene="data/" \
		--test_transform="data/new_render/metadata.json"\
		--screenshot_frequency=500\
		--n_steps=2000\
		--width=720\
		--height=1080\
		--screenshot_transforms="data/new_render/metadata.json"\
		--screenshot_dir="chenhao_lab/output"
