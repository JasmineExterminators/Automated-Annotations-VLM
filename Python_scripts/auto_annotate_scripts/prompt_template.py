def get_prompt(task_name: str, context: str, time_gap: float) -> str:
    return f"""You are a robot performing the task {task_name}. You are given two files. 
    
    1. The first frame of the video displaying the initial state of the scene.                    
    2. The previous frame from {time_gap} seconds ago. 
    3. The current frame that depicts the current state of the scene. 
    4. The previous context: {context}

The name of the frame ("previous", "current", or "first") is provided in the top right corner of the image.

WARNING: The previous context is not necessarily accurate. The context DOES NOT gurantee that a previous action has actually been completed. DO NOT depend solely on the previous context. Focus on the current frame and the observations you can make from it.                      

The left side shows the front view and the right side shows the view on the grippers of the robot. 

MISSION: Your mission is to generate a detailed action and reasoning for the robot to take in the current frame.

1. Examine the previous and current frame. Infer what happened between the two frames and what is happening right now. When observing, pay careful attention to the task name, {task_name}. Note object spatial relationships and the robot position. 
                                            
2. Think about your observations and the past context. Generate an action that the robot should take in the current frame as well as a detailed reasoning for the action. The action annotation is very fine-grained. For example, grasping is divided into 2 actions: reach, close the gripper. 

If the task description highlights spatial relationships, or if there are multiple objects from the same category, then your action and reasoning should also contain these spatial / directional info, such as left / right, front / back. Focus on key visual features that help you identify the current situation. For example, the robot "is holding something." or "has not reached something." For instance, "lift the bowl upwards and to the left towards the stove."

REMEMBER: the task name is {task_name}. 

An example action is "reach for the black bowl by the white plate." An example reasoning is "I am reaching for the black bowl by the white plate because I need to pick it up and place it in the caddy. The bowl is on the left side of the plate, and I need to ensure I grasp it securely."

IMPORTANT: It is imperative that you do not hallucinate actions or reasonings. For instance, closely examine the eye-in-hand view of the robot. If the robot is not grasping an object, do not annotate it as such. It is okay for intervals to be annotated similarly or to annotate intermediate actions like "continue holding" or "maintain position" if the robot is not performing a distinct action.
""" 