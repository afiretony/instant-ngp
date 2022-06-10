#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import commentjson as json
import matplotlib.pyplot as plt
import json
import numpy as np

import sys
import time

from common import *
from scenes import scenes_nerf, scenes_image, scenes_sdf, scenes_volume, setup_colored_sdf

from tqdm import tqdm
# pyngp_path = 'build'
# sys.path.append(pyngp_path)
# import pyngp as ngp
import pyngp as ngp # noqa


def parse_args():
	parser = argparse.ArgumentParser(description="Run neural graphics primitives testbed with additional configuration & output options")

	parser.add_argument("--scene", "--training_data", default="", help="The scene to load. Can be the scene's name or a full path to the training data.")
	parser.add_argument("--mode", default="", const="nerf", nargs="?", choices=["nerf", "sdf", "image", "volume"], help="Mode can be 'nerf', 'sdf', or 'image' or 'volume'. Inferred from the scene if unspecified.")
	parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")

	parser.add_argument("--load_snapshot", default="", help="Load this snapshot before training. recommended extension: .msgpack")
	parser.add_argument("--save_snapshot", default="", help="Save this snapshot after training. recommended extension: .msgpack")

	parser.add_argument("--nerf_compatibility", action="store_true", help="Matches parameters with original NeRF. Can cause slowness and worse results on some scenes.")
	parser.add_argument("--test_transforms", default="", help="Path to a nerf style transforms json from which we will compute PSNR.")
	parser.add_argument("--near_distance", default=-1, type=float, help="set the distance from the camera at which training rays start for nerf. <0 means use ngp default")

	parser.add_argument("--screenshot_transforms", default="", help="Path to a nerf style transforms.json from which to save screenshots.")
	parser.add_argument("--screenshot_cams", nargs="*", help="Which frame(s) to take screenshots of.")
	parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots to.")
	parser.add_argument("--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots.")
	parser.add_argument("--screenshot_frequency", default=1000,type=int, help="Frequency of screenshot of test and training transforms")

	parser.add_argument("--save_mesh", default="", help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.")
	parser.add_argument("--marching_cubes_res", default=256, type=int, help="Sets the resolution for the marching cubes grid.")

	parser.add_argument("--width", "--screenshot_w", type=int, default=0, help="Resolution width of GUI and screenshots.")
	parser.add_argument("--height", "--screenshot_h", type=int, default=0, help="Resolution height of GUI and screenshots.")

	parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
	parser.add_argument("--train", action="store_true", help="If the GUI is enabled, controls whether training starts immediately.")
	parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")

	parser.add_argument("--sharpen", default=0, help="Set amount of sharpening applied to NeRF training images.")
	parser.add_argument("--work_space", type=str, help="Set the work dir to save training results etc.")
	

	args = parser.parse_args()
	return args

def render_frames(args, ref_transforms, out_folder, step):
	
	if not os.path.exists(out_folder):
		os.mkdir(out_folder)

	testbed.fov_axis = 0
	testbed.fov = ref_transforms["camera_angle_x"] * 180 / np.pi

	for idx in range(len(ref_transforms["frames"])):
		f = ref_transforms["frames"][int(idx)]
		cam_matrix = f["transform_matrix"]
		testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1,:])
		outname = os.path.join('cam_{:03d}_{:05d}.png'.format(idx, step))

		print(f"rendering {outname}")
		image = testbed.render(args.width or int(ref_transforms["w"]), args.height or int(ref_transforms["h"]), args.screenshot_spp, True)
		write_image(os.path.join(out_folder, outname), image)

def eval(mode, step, transforms):
	print("Evaluating"+mode+ " transforms from ", transforms)
	with open(transforms) as f:
		test_transforms = json.load(f)
	data_dir=os.path.dirname(transforms)
	totmse = 0
	totpsnr = 0
	totssim = 0
	totcount = 0
	minpsnr = 1000
	maxpsnr = 0

	# Evaluate metrics on black background
	testbed.background_color = [0.0, 0.0, 0.0, 1.0]

	# Prior nerf papers don't typically do multi-sample anti aliasing.
	# So snap all pixels to the pixel centers.
	testbed.snap_to_pixel_centers = True
	spp = 8

	testbed.nerf.rendering_min_transmittance = 1e-4

	testbed.fov_axis = 0
	testbed.fov = test_transforms["camera_angle_x"] * 180 / np.pi
	# testbed.shall_train = False

	with tqdm(list(enumerate(test_transforms["frames"])), unit="images", desc=f"Rendering test frame") as t:
		for i, frame in t:
			p = frame["file_path"]
			if "." not in p:
				p = p + ".png"
			ref_fname = os.path.join(data_dir, p)
			if not os.path.isfile(ref_fname):
				ref_fname = os.path.join(data_dir, p + ".png")
				if not os.path.isfile(ref_fname):
					ref_fname = os.path.join(data_dir, p + ".jpg")
					if not os.path.isfile(ref_fname):
						ref_fname = os.path.join(data_dir, p + ".jpeg")
						if not os.path.isfile(ref_fname):
							ref_fname = os.path.join(data_dir, p + ".exr")

			ref_image = read_image(ref_fname)

			# NeRF blends with background colors in sRGB space, rather than first
			# transforming to linear space, blending there, and then converting back.
			# (See e.g. the PNG spec for more information on how the `alpha` channel
			# is always a linear quantity.)
			# The following lines of code reproduce NeRF's behavior (if enabled in
			# testbed) in order to make the numbers comparable.
			if testbed.color_space == ngp.ColorSpace.SRGB and ref_image.shape[2] == 4:
				# Since sRGB conversion is non-linear, alpha must be factored out of it
				ref_image[...,:3] = np.divide(ref_image[...,:3], ref_image[...,3:4], out=np.zeros_like(ref_image[...,:3]), where=ref_image[...,3:4] != 0)
				ref_image[...,:3] = linear_to_srgb(ref_image[...,:3])
				ref_image[...,:3] *= ref_image[...,3:4]
				ref_image += (1.0 - ref_image[...,3:4]) * testbed.background_color
				ref_image[...,:3] = srgb_to_linear(ref_image[...,:3])

			
			# write_image("ref.png", ref_image)

			testbed.set_nerf_camera_matrix(np.matrix(frame["transform_matrix"])[:-1,:])
			image = testbed.render(ref_image.shape[1], ref_image.shape[0], spp, True)

			out_folder = os.path.join(args.work_space, "checkpoint_{:05d}".format(step), mode+"_res")
			if not os.path.exists(out_folder):
				# os.mkdir(out_folder)
				os.makedirs(out_folder, mode = 0o777, exist_ok = False) 
			out_name = os.path.join(out_folder, "RGBA_CAM_{:03d}.png".format(i))
			write_image(out_name, image)

			diffimg = np.absolute(image - ref_image)
			diffimg[...,3:4] = 1.0
			# if i == 0:
			# 	write_image("diff.png", diffimg)

			A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
			R = np.clip(linear_to_srgb(ref_image[...,:3]), 0.0, 1.0)
			mse = float(compute_error("MSE", A, R))
			ssim = float(compute_error("SSIM", A, R))
			totssim += ssim
			totmse += mse
			psnr = mse2psnr(mse)
			totpsnr += psnr
			minpsnr = psnr if psnr < minpsnr else minpsnr
			maxpsnr = psnr if psnr > maxpsnr else maxpsnr
			totcount = totcount + 1
			t.set_postfix(psnr = totpsnr/(totcount or 1))

	psnr_avgmse = mse2psnr(totmse/(totcount or 1))
	psnr = totpsnr/(totcount or 1)
	ssim = totssim/(totcount or 1)
	print(f"PSNR={psnr} [min={minpsnr} max={maxpsnr}] SSIM={ssim}")
	return psnr, ssim, maxpsnr, minpsnr


if __name__ == "__main__":
	args = parse_args()


	mode = ngp.TestbedMode.Nerf
	configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")

	base_network = os.path.join(configs_dir, "base.json")
	network = args.network if args.network else base_network

	if not os.path.isabs(network):
		network = os.path.join(configs_dir, network)

	testbed = ngp.Testbed(mode)
	testbed.nerf.sharpen = float(args.sharpen)

	train_transforms = os.path.join(args.scene, 'transforms_train_config_1.json') 
	test_transforms = os.path.join(args.scene, 'transforms_test_focus.json')

	testbed.load_training_data(train_transforms)
	testbed.reload_network_from_file(network)
	print("Training loaded")

	testbed.shall_train =  True

	testbed.nerf.render_with_camera_distortion = True

	network_stem = os.path.splitext(os.path.basename(network))[0]

	if args.near_distance >= 0.0:
		print("NeRF training ray near_distance ", args.near_distance)
		testbed.nerf.training.near_distance = args.near_distance

	old_training_step = 0
	n_steps = args.n_steps
	if n_steps < 0:
		n_steps = 10000

	log = {}
	log["psnr_train"]= {}
	log["psnr_test"] = {}
	log["ssim_train"]= {}
	log["ssim_test"] = {}
	log["loss"] = {}
	log["time"] = {}
	loss_frequency = 10

	start_time = time.time()
	rendering_time = 0.0 # to subtract


	if n_steps > 0:
		with tqdm(desc="Training", total=n_steps, unit="step") as t:
			while testbed.frame():

				if testbed.want_repl():
					repl(testbed)
				# What will happen when training is done?
				if testbed.training_step >= n_steps:
					if args.gui:
						testbed.shall_train = False
					else:
						break

				# Update progress bar
				if testbed.training_step < old_training_step or old_training_step == 0:
					old_training_step = 0
					t.reset()

				t.update(testbed.training_step - old_training_step)
				t.set_postfix(loss=testbed.loss)
				old_training_step = testbed.training_step

				curr_time = time.time()-start_time-rendering_time

				if testbed.training_step % loss_frequency == 0:
					log["loss"][testbed.training_step] = testbed.loss

				# render and save training frame results
				if testbed.training_step % args.screenshot_frequency == 0:
					print("Rendering results")
					temp_time = time.time()
					out_folder = os.path.join(args.work_space, 'training_output')
					# render_frames(args, ref_transforms, out_folder, testbed.training_step)
					
					train_psnr, train_ssim, train_maxpsnr, train_minpsnr = eval("train", testbed.training_step, train_transforms)
					test_psnr, test_ssim, test_maxpsnr, test_minpsnr = eval("test", testbed.training_step, test_transforms)
					log["psnr_test"][testbed.training_step]=test_psnr
					log["ssim_test"][testbed.training_step]=test_ssim
					log["psnr_train"][testbed.training_step]=train_psnr
					log["ssim_train"][testbed.training_step]=train_ssim
					log["time"][testbed.training_step]=curr_time
					rendering_time += time.time()-temp_time

	with open(os.path.join(args.work_space, 'results.json'),'w') as outfile:
		json.dump(log, outfile, indent=4)

	# if args.save_snapshot:
	# 	print("Saving snapshot ", args.save_snapshot)
	# 	testbed.save_snapshot(args.save_snapshot, False)


	if args.save_mesh:
		res = args.marching_cubes_res or 512
		print(f"Generating mesh via marching cubes and saving to {args.save_mesh}. Resolution=[{res},{res},{res}]")
		testbed.compute_and_save_marching_cubes_mesh(os.path.join(args.work_space, args.save_mesh), [res, res, res])



	# with open("test_config_1/results.json") as f:
	# 	log = json.load(f)

	plt.figure(0)
	title = 'Training loss'
	plt.plot(log["loss"].keys(),log["loss"].values(), label='training loss')
	plt.xlabel("n iteration")
	plt.title(title)
	plt.legend(loc="upper right")
	plt.savefig(os.path.join(args.work_space, title+'.png'))
	
	plt.figure(1)
	title = "PSNR"
	plt.plot(log["psnr_test"].keys(), log["psnr_test"].values(), label="test")
	plt.plot(log["psnr_train"].keys(), log["psnr_train"].values(), label="train")
	plt.xlabel("n iteration")
	plt.title(title)
	plt.grid('both')
	plt.legend(loc="upper right")
	plt.savefig(os.path.join(args.work_space, title+'.png'))

	
	plt.figure(2)
	title = "SSIM"
	plt.plot(log["ssim_test"].keys(), log["ssim_test"].values(), label="test")
	plt.plot(log["ssim_train"].keys(), log["ssim_train"].values(), label="train")
	plt.xlabel("n iteration")
	plt.title(title)
	plt.grid('both')
	plt.legend(loc="upper right")
	plt.savefig(os.path.join(args.work_space, title+'.png'))
	
