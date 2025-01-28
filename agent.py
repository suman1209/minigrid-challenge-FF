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
        self.past_states = deque(maxlen=20)
        self.current_step = 0

        # System prompt to explain the task
        self.system_prompt = """You are an agent in a grid-world environment. You will receive information about:
1. The direction you are facing
2. Your current mission
3. Objects you can see in front of you
4. Whether you are touching a wall
4. Past states and taken actions

You must choose one of these actions:
- turn left (rotates counterclockwise)
- turn right (rotates clockwise)
- move forward
- pick up
- drop
- toggle

Additional information:
- You cannot step on objects, you need to avoid them
- Turning left or right will change your field of view
- Locked doors can be toggled with a key, if they are one cell in front of you
- Keys can be picked up and dropped
- Box can contain a key or another object
- Box can be toggled to reveal its content if it's one cell in front of you
- If you don't see target object, move around to find it"""

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
        direction = directions[obs["direction"]]

        # Parse the grid to find visible objects
        visible_objects = []
        grid = obs["image"]

        # Convert object types to descriptions
        for x in range(7):
            for y in range(7):
                obj_id, color_id, door_state = grid[x, y]
                if obj_id > 2:
                    obj_state = ""
                    if obj_id == 4:  # it's a door
                        obj_state = f"{IDX_TO_STATE[door_state]} "
                    msg = f"\n * {obj_state}{IDX_TO_COLOR[color_id]} {IDX_TO_OBJECT[obj_id]} -"
                    if x < 3:
                        msg += f" {3 - x} cells to the {relative_to_absolute(direction, 'left')};"
                    elif x > 3:
                        msg += f" {x - 3} cells to the {relative_to_absolute(direction, 'right')};"
                    if y < 6:
                        msg += f" {6 - y} cells to the {relative_to_absolute(direction, 'front')};"
                    visible_objects.append(msg)
        actionable_object = "none"
        if grid[3, 5, 0] > 2:
            actionable_object = (
                f"{IDX_TO_COLOR[grid[3, 6, 1]]} {IDX_TO_OBJECT[grid[3, 6, 0]]}"
            )

        walls = []
        if grid[2, 6, 0] == 2:
            walls.append(relative_to_absolute(direction, "left"))
        if grid[4, 6, 0] == 2:
            walls.append(relative_to_absolute(direction, "right"))
        if grid[3, 5, 0] == 2:
            walls.append(relative_to_absolute(direction, "front"))
        if len(walls) == 0:
            walls.append("none")

        # Create the prompt
        past_states_str = "\n".join(self.past_states)
        current_state = f"""[Step {self.current_step}]
- Facing '{direction}'
- Visible objects: {', '.join(visible_objects) if visible_objects else 'none'}
- Actionable object: {actionable_object}
- Touching walls: {walls}"""
        prompt = f"""Mission: {mission}

Past states:
{past_states_str}

Current state:
{current_state}

What action should you take? Respond ONLY with the action you want to take, exactly as written above."""

        return prompt, current_state

    def get_action(self, obs: Dict[str, Any], mission: str, verbose: bool) -> int:
        """
        Get the next action from the model.

        Args:
            obs: Observation from the environment
            mission: Current mission string

        Returns:
            Action index
        """
        prompt, current_state = self.parse_observation(obs, mission)
        final_prompt = f"{self.system_prompt}\n\n{prompt}"

        if verbose:
            print("==================================")
            print("final_prompt:", final_prompt)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": final_prompt},
            ],
            temperature=self.temperature,
            max_tokens=10,
        )
        if verbose:
            print("Response:", response.choices[0].message.content)

        action_text = response.choices[0].message.content.strip().lower()
        if "\n" in action_text:
            action_text = action_text.split("\n")[0]

        # Convert action text to action index
        action_idx = None
        for idx, text in ACTION_MAP.items():
            if text == action_text:
                action_idx = idx
                break

        if action_idx is None:
            print(
                f"Warning: Invalid action '{action_text}', defaulting to move forward"
            )
            action_idx = 2  # Default to move forward
            action_text = ACTION_MAP[2]

        self.past_states += [
            current_state,
            f"Action taken at step {self.current_step}: {action_text}",
        ]
        self.current_step += 1

        return action_idx


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

    action = agent.get_action(obs, mission, verbose)

    if verbose:
        print("Chosen Action:", ACTION_MAP[action])

    return action
