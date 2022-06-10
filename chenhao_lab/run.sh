#!usr/bin/bash
python chenhao_lab/run.py --mode="nerf" \
                --scene="data/dev-ch" \
		--screenshot_frequency=500\
		--n_steps=2501\
		--width=720\
		--height=1080\
		--work_space="chenhao_lab/output/test_config_4"\
		--save_mesh="generated_mesh.obj"\
		--marching_cubes_res=512
