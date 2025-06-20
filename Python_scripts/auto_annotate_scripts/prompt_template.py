def get_annotate_prompt(task_name: str, context: str, time_gap: float, task_list: list[str], first, seconds_passed) -> str:
    prompt = f"""You are a robot performing the task {task_name}. You are given three files. 
    
    1. The previous frame from {time_gap} seconds ago. 
    2. The current frame that depicts the current state of the scene. 
    3. A live, suggested to-do list of subtasks that make up the full task {task_list if first else context}.  

The name of the frame ("previous" or "current") is provided in the top right corner of the image. It has been {seconds_passed} seconds since the robot started moving.               

The left side shows the front view and the right side shows the view on the grippers of the robot. IMPORTANT: Pay careful attention to the right eye-in-hand view. If the eye-in-hand view shows that the robot is not grasping something, then the robot is not grasping anything. Only write an action to close the gripper if the eye-in-hand camera shows that there is an object in-between the grippers, not just near the grippers.

MISSION: Your mission is to generate a detailed action and reasoning for the robot to take in the current frame.

1. Examine the previous and current frame. Infer what happened between the two frames and what is happening right now. When observing, pay careful attention to the task name, {task_name}. Note object spatial relationships and the robot position. 

2. Look at the task list. If the next task on the list appears to be completed, cross it off of the list. Then, scan the rest of the tasks in the list. Are any of them no longer relevant? Carefully modify the list if necessary. When unsure, leave the list alone. Return the list in the "summary" field.

If the task list does not match up with your current observations, modify the task list. 

IMPORTANT: At any timestep, it is not guranteed that a task will be completed. Pay careful attention to the video to determine if task should or should not be crossed off yet.
                                            
3. Think about your observations and the task list. Generate an action that the robot should take in the current frame as well as a detailed reasoning for the action. The action annotation is very fine-grained. For example, grasping is divided into 2 actions: reach, close the gripper. 

REMEMBER: the task name is {task_name}. 

An example action is "reach for the black bowl by the white plate." An example reasoning is "I am reaching for the black bowl by the white plate because I need to pick it up and place it in the caddy. The bowl is on the left side of the plate, and I need to ensure I grasp it securely."
""" 

    print(f"""{task_list if first else context}""")
    return prompt

# If the task description highlights spatial relationships, or if there are multiple objects from the same category, then your action and reasoning should also contain these spatial / directional info, such as left / right, front / back. Focus on key visual features that help you identify the current situation. For example, the robot "is holding something." or "has not reached something." For instance, "lift the bowl upwards and to the left towards the stove."


def get_task_prompt(task_name: str) -> str:
    return f"""
        You are a robot assigned the following task: {task_name}. You are also given a frame of the current scene. 
        
        MISSION: Generate a task list of actions the robot should perform in order to successfully complete the task. Task location refers to where in the scene the robot should perform the task, could involve multiple locations. Task duration should be in seconds. 
        
        Some tips:
            1. The order of the sub-tasks in the task name: {task_name} is the order the subtasks in {task_name} should be performed. 
            2. Provide detailed, actionable, fine-grained steps. For instance, grasping an object should be split into two actions: reach for the object, lower the gripper, and close the gripper around the object.
    """