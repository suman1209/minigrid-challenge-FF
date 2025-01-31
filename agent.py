from openai import OpenAI
from typing import Dict, Any, Tuple, Optional
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR, STATE_TO_IDX
from collections import deque

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}

ACTION_MAP = {
    0: "turn left",
    1: "turn right",
    2: "move forward",
    3: "pick up",
    4: "drop",
    5: "toggle",
    6: "done",
}


def relative_to_absolute(agent_direction, relative_direction):
    if agent_direction == "north":
        if relative_direction == "left":
            return "west"
        elif relative_direction == "right":
            return "east"
        elif relative_direction == "front":
            return "north"
    elif agent_direction == "south":
        if relative_direction == "left":
            return "east"
        elif relative_direction == "right":
            return "west"
        elif relative_direction == "front":
            return "south"
    elif agent_direction == "east":
        if relative_direction == "left":
            return "north"
        elif relative_direction == "right":
            return "south"
        elif relative_direction == "front":
            return "east"
    elif agent_direction == "west":
        if relative_direction == "left":
            return "south"
        elif relative_direction == "right":
            return "north"
        elif relative_direction == "front":
            return "west"
    else:
        raise ValueError(f"Invalid agent direction: {agent_direction}")


class Agent:
    def __init__(
        self, api_key: str, model: str = "gpt-4o-mini", api_url: Optional[str] = None
    ):
        """
        Initialize the agent.

        Args:
            api_key: API key
            model: model to use
            temperature: Temperature for model sampling
        """
        self.client = OpenAI(api_key=api_key, base_url=api_url)
        self.model = model
        self.temperature = 0.0
        self.past_states = deque(maxlen=2)  # [state, response]
        self.current_step = 0
        self.past_actions = []
        # System prompt to explain the task
    # @todo why do we do this?
    def find_last_action(self, action_text, action_map):
        action_idx = None
        last_position = -1
        found_action = None

        # Check each possible action
        for idx, text in action_map.items():
            # Find the last position of this action in the text
            position = action_text.rfind(text)

            # If found and it's later than our previous match
            if position != -1 and position > last_position:
                last_position = position
                action_idx = idx
                found_action = text

        return action_idx, found_action

    def get_system_prompt(self, direction):
        """
        Generates a system prompt for an agent in a grid-world environment.

        Parameters:
        direction (str): The current direction the agent is facing (e.g., "north", "south", etc.).

        Returns:
        str: A formatted prompt explaining the agent's objectives and rules.
        """
        return f"""You are an agent in a grid-world environment. The goal is to navigate the world and interact with objects to complete the mission.

    You must ONLY choose one of these actions:
    - turn left (rotates towards {relative_to_absolute(direction, 'left')})
    - turn right (rotates towards {relative_to_absolute(direction, 'right')})
    - move forward (moves towards {direction})
    - pick up (grabs an object directly in front of you)
    - drop (places a carried object on the floor)
    - toggle (opens a door with a key or opens a box)
    
    - DO NOT suggest the same action from the previous step that did not result in a change in state. If it is necessary, a proper explanation is important.

    Movement and Interaction Rules:
    - You can face FOUR different directions: north, south, east, west
    - You cannot step on objects; you need to go around them
    - Objects include: box, ball, key
    - Locked doors can be toggled with a key, if they are one cell in front of you
    - Keys can be picked up and used to unlock doors
    - A box can contain a key or another object
    - A box can be toggled to reveal its content
    - Decide if you need to pickup or toggle the box depending on the task
    - You can pick up and toggle only actionable objects
    - The color of the object in the mission is important for deciding movement.
    - After unlocking a door, drop the key.
    - If you don't see the target object, explore the world to find it.
    - If you turn right or left, you may lose the object from your sight, so you need to remember where it was.

    Decision Making:
    - If your previous action had no effect, try a different approach.
    - If the target object is not visible, explore the world until you find it.
    - Remember object locations, even when they are out of sight.
    - If turning left or right removes an object from sight, recall its position.

    Example Walkthrough:
    For example, if I am facing north and the mission is to pick up the yellow box that is two cells to the right and one cell in front, then the sequence of actions will look like:
    ["turn right", "move forward", "move forward", "turn left", "move forward", "pick up", "drop"].

    Your Task:
    1. Plan the next best action based on the mission and environment.
    2. Think step-by-step, but provide only the next action.

    What action should you take? Provide a reason for the choice but in the last line, provide ONLY the next best action you want to take, exactly as written above."""

    def parse_observation(self, obs: Dict[str, Any], mission: str) -> str:
        """
        Convert the observation into a text prompt for the model.

        Args:
            obs: Observation from the environment
            mission: Current mission string

        Returns:
            Formatted prompt string
        """
        # Convert direction number to cardinal direction
        directions = ["east", "south", "west", "north"]
        # @todo, check the observations
        direction = directions[obs["direction"]]

        # Parse the grid to find visible objects
        visible_objects = []
        grid = obs["image"]

        # Convert object types to descriptions
        for x in range(7):
            for y in range(7):
                if x == 3 and y == 6:
                    continue  # skip for agent position - it's the object being held
                obj_id, color_id, door_state = grid[x, y]
                """
                COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
                OBJECT_TO_IDX = {"unseen": 0, "empty": 1, "wall": 2, "floor": 3, "door": 4, "key": 5, "ball": 6, 
                                 "box": 7, "goal": 8, "lava": 9, "agent": 10}

                STATE_TO_IDX = { 
                                "open": 0, 
                                "closed": 1,
                                "locked": 2
                                }
                """
                if obj_id > 2:
                    obj_state = ""
                    if obj_id == 4:  # it's a door
                        obj_state = f"{IDX_TO_STATE[door_state]} "
                    obj_repr = f"\n * {obj_state}{IDX_TO_COLOR[color_id]} {IDX_TO_OBJECT[obj_id]} -"
                    obj_pos = ""
                    if x < 3:
                        obj_pos += f" {3 - x} cells to the left"
                    elif x > 3:
                        obj_pos += f" {x - 3} cells to the right"
                    if y < 6:
                        if obj_pos != "":
                            obj_pos += " AND"
                        obj_pos += f" {6 - y} cells in the front"
                    obj_repr = obj_repr + obj_pos
                    visible_objects.append(obj_repr)

        actionable_object = "none"
        if grid[3, 5, 0] > 2:
            actionable_object = (
                f"{IDX_TO_COLOR[grid[3, 5, 1]]} {IDX_TO_OBJECT[grid[3, 5, 0]]}"
            )
        holding_object = "none"
        if grid[3, 6, 0] > 2:
            holding_object = (
                f"{IDX_TO_COLOR[grid[3, 6, 1]]} {IDX_TO_OBJECT[grid[3, 6, 0]]}"
            )

        walls = [0, 0, 0]
        if True:
            for i in range(3):
                if grid[i, 6, 0] == 2:
                    walls[0] = 3 - i
                    break
        if True:
            for i in range(4, 7):
                if grid[i, 6, 0] == 2:
                    walls[1] =  i - 4
                    break
        if True:
            for i in range(5, -1, -1):
                if grid[3, i, 0] == 2:
                    walls[2] = 5 - i
                    break
        if len(walls) == 0:
            walls.append("none")

        # Create the prompt
        past_states_str = "\n".join(self.past_states)
        current_state = f"""[Step {self.current_step}]
- Facing '{direction}'
- Wall on the left: {str(walls[0]) + " cells further" if walls[0] > 0 else "no"}
- Wall on the right: {str(walls[1]) + " cells further" if walls[1] > 0 else "no"}
- Wall in front: {str(walls[2]) + " cells further" if walls[2] > 0 else "no"}
- Visible objects: {', '.join(visible_objects) if visible_objects else 'none'}
- Actionable object: {actionable_object}
- Holding object: {holding_object}
- Mission: {mission}"""
        prompt = f"""Below is the actions taken and the observations from the previous steps, please consider what action
        was taken previously and what the response was in determining the next best action:
{past_states_str}
{current_state}
Action taken: {self.past_actions[-2:]}"""

        return prompt, current_state, direction

    def get_action(self, obs: Dict[str, Any], mission: str, verbose: bool) -> int:
        """
        Get the next action from the model.

        Args:
            obs: Observation from the environment
            mission: Current mission string

        Returns:
            Action index
        """
        prompt, current_state, direction = self.parse_observation(obs, mission)
        final_prompt = f"{self.get_system_prompt(direction)}\n\n{prompt}"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": final_prompt},
            ],
            temperature=self.temperature,
            max_tokens=1000,
        )
        if verbose:
            print("==================================")
            print("final_prompt:\n", final_prompt)
            print("response:\n", response.choices[0].message.content)

        response = response.choices[0].message.content.strip().lower()

        action_idx, action_text = self.find_last_action(response, ACTION_MAP)
        self.past_actions.append(action_text)
        if action_idx is None:
            print(
                f"Warning: Invalid action '{action_text}', defaulting to move forward"
            )
            print(action_idx, action_text, response)
            action_idx = 2  # Default to move forward
            action_text = ACTION_MAP[2]

        self.past_states += [
            current_state,
            f"Response: {action_text}",
        ]
        self.current_step += 1

        # dict with metadata to log during eval
        metadata = {
            "final_prompt": final_prompt,
            "response": response,
            "action_text": action_text,
        }

        return action_idx, metadata


def handle_state(
    obs: Dict[str, Any], mission: str, agent: Agent, verbose: bool = False
) -> int:
    """
    Process the current state and get the next action.

    Args:
        obs: Current observation from the environment
        mission: Current mission string
        agent: Agent instance
        verbose: Whether to print debug information

    Returns:
        Action index to take
    """

    action, metadata = agent.get_action(obs, mission, verbose)

    if verbose:
        print("Chosen Action:", ACTION_MAP[action])

    return action, metadata
