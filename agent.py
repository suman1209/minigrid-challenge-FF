from openai import OpenAI
from typing import Dict, Any, Tuple, Optional
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR, STATE_TO_IDX
from collections import deque


ACTION_MAP = {
    0: "turn left",
    1: "turn right",
    2: "move forward",
    3: "pick up",
    4: "drop",
    5: "toggle",
    6: "done",
}


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
        self.last_actions = deque(maxlen=5)

        # System prompt to explain the task
        self.system_prompt = """
        You are an agent in a grid-world environment. You will receive information about:
        1. The direction you are facing
        2. Your current mission
        3. Objects you can see in front of you
        4. Whether you are touching a wall
        4. Last taken actions
        
        You must choose one of these actions:
        - turn left
        - turn right
        - move forward
        - pick up
        - drop
        - toggle

        Additional information:
        - You cannot step on objects, you need to avoid them
        - If you've been doing the same action multiple times, try something else
        - Turning left or right will change your field of view
        
        Respond ONLY with the action you want to take, exactly as written above.
        """

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
        directions = ["right", "down", "left", "up"]
        direction = directions[obs["direction"]]

        # Parse the grid to find visible objects
        visible_objects = []
        grid = obs["image"]

        # Convert object types to descriptions
        for x in range(7):
            for y in range(7):
                obj_id, color_id, door_state = grid[x, y]
                if obj_id > 2:
                    msg = f"\n * {IDX_TO_COLOR[color_id]} {IDX_TO_OBJECT[obj_id]} -"
                    if x < 3:
                        msg += f" {3 - x} cells to the left;"
                    elif x > 3:
                        msg += f" {x - 3} cells to the right;"
                    if y < 6:
                        msg += f" {6 - y} cells in front of you;"
                    visible_objects.append(msg)

        walls = []
        if grid[2, 6, 0] == 2:
            walls.append("left")
        if grid[4, 6, 0] == 2:
            walls.append("right")
        if grid[3, 5, 0] == 2:
            walls.append("front")
        if len(walls) == 0:
            walls.append("none")

        # Create the prompt
        prompt = f"""
        Mission: {mission}
        
        Current state:
        - You are facing '{direction}'
        - Objects in front of you: {', '.join(visible_objects) if visible_objects else 'none'}
        - Touching walls: {walls}
        - Last action: {list(self.last_actions) if self.last_actions else 'none'}
        
        What action should you take?
        """

        return prompt

    def get_action(self, obs: Dict[str, Any], mission: str, verbose: bool) -> int:
        """
        Get the next action from the model.

        Args:
            obs: Observation from the environment
            mission: Current mission string

        Returns:
            Action index
        """
        prompt = f"{self.system_prompt}\n\n{self.parse_observation(obs, mission)}"
        if verbose:
            print("==================================")
            print("Prompt:", prompt)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
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
                self.last_actions.append(action_text)
                action_idx = idx
                break

        if action_idx is None:
            print(
                f"Warning: Invalid action '{action_text}', defaulting to move forward"
            )
            action_idx = 2  # Default to move forward

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
