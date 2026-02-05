# import packages and module here
import sys, os
from .model import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def encode_obs(observation):  # Post-Process Observation
    observation["agent_pos"] = observation["joint_action"]["vector"]
    return observation


def get_model(usr_args):  # keep
    model_name = usr_args["ckpt_setting"]
    checkpoint_id = usr_args["checkpoint_id"]
    left_arm_dim, right_arm_dim, rdt_step = (
        usr_args["left_arm_dim"],
        usr_args["right_arm_dim"],
        usr_args["rdt_step"],
    )
    
    use_ema = usr_args.get("use_ema", False)
    if use_ema:
        main_checkpoint_path = os.path.join(
            parent_directory,
            f"checkpoints/{model_name}/checkpoint-{checkpoint_id}",
        )
        ema_checkpoint_path = os.path.join(
            parent_directory,
            f"checkpoints/{model_name}/checkpoint-{checkpoint_id}/ema",
        )
        print(f"Loading EMA model from: {ema_checkpoint_path}")
        print(f"Using config from: {main_checkpoint_path}")
        checkpoint_path = main_checkpoint_path
        usr_args["ema_model_path"] = ema_checkpoint_path
    else:
        checkpoint_path = os.path.join(
            parent_directory,
            f"checkpoints/{model_name}/checkpoint-{checkpoint_id}",
        )
        print(f"Loading main model from: {checkpoint_path}")
    
    rdt = RDT(
        checkpoint_path,
        usr_args["task_name"],
        left_arm_dim,
        right_arm_dim,
        rdt_step,
    )
    if use_ema:
        rdt.ema_model_path = usr_args["ema_model_path"]
    
    return rdt


def eval(TASK_ENV, model, observation):
    """x
    All the function interfaces below are just examples
    You can modify them according to your implementation
    But we strongly recommend keeping the code logic unchanged
    """
    obs = encode_obs(observation)  # Post-Process Observation
    instruction = TASK_ENV.get_instruction()
    input_rgb_arr, input_state = [
        obs["observation"]["head_camera"]["rgb"],
        obs["observation"]["right_camera"]["rgb"],
        obs["observation"]["left_camera"]["rgb"],
    ], obs["agent_pos"]  # TODO

    if (model.observation_window
            is None):  # Force an update of the observation at the first frame to avoid an empty observation window
        model.set_language_instruction(instruction)
        model.update_observation_window(input_rgb_arr, input_state)

    actions = model.get_action()  # Get Action according to observation chunk

    for action in actions:  # Execute each step of the action
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        input_rgb_arr, input_state = [
            obs["observation"]["head_camera"]["rgb"],
            obs["observation"]["right_camera"]["rgb"],
            obs["observation"]["left_camera"]["rgb"],
        ], obs["agent_pos"]  # TODO
        model.update_observation_window(input_rgb_arr, input_state)  # Update Observation


def reset_model(
        model):  # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.reset_obsrvationwindows()
