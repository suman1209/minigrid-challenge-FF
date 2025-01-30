# MiniGrid Agent Challenge

Your task is to create an agent that can solve different grid-world environments using language models. The agent will be evaluated on 10 different environments - 6 provided for development and 4 hidden for final evaluation - all 10 will used for the final score.

### Evaluation Environments (Provided)

-   BabyAI-GoToLocalS8N3-v0
-   BabyAI-KeyInBox-v0
-   BabyAI-PutNextS5N2-v0
-   BabyAI-UnlockPickup-v0
-   BabyAI-KeyCorridorS3R2-v0
-   BabyAI-UnlockToUnlock-v0

### Rules

1. You can only modify the `agent.py` file
2. Do not change the signature of the `Agent` constructor or `handle_state` function
3. Any local model can be used, as long as it runs on an A40 GPU
4. If the model runs out of memory during evaluation, the episode is lost - make sure you handle context window properly.
5. Each environment will be evaluated with 4 episodes - maximum 100 steps per episode, with 6-minutes timeout per episode
6. During final evaluation, we will use a different RNG seed than the default value
7. Environment is considered to be solved if there's at least 2 wins over 4 episodes
8. Winner is determined by the number of solved environments - in case of a draw, the total accuracy determines the winner
9. Do NOT explicitly code the action logic - e.g. "toggle" IF chest in front of you
10. You can do multiple calls to the language model per step
11. Use of LLMs for development is allowed - try Claude with Sonnet 3.5, or Cursor IDE with different models

## Prerequisites

-   Access to Alex cluster (configure SSH settings on the local machine to access with `ssh alex` for convenience)
-   Ability to run A40 GPU jobs
-   Miniconda/Python installed

## Setup

1. SSH into the cluster:

```bash
ssh alex
```

2. Install dependencies

```bash
pip install minigrid openai vllm
```

3. Request an A40 GPU:

```bash
salloc --gpus-per-node=1 --partition=a40 --time=4:00:00
```

4. Start the model server (example with Qwen):

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8081
```

5. On your local machine, create SSH tunnel (replace `a0322` with your assigned node), and keep the terminal open:

```bash
node=a0322
ssh -L 8081:$node:8081 alex
```

6. Start the script (example with Qwen and BabyAI-GoToLocalS8N3-v0 env)

```bash
python main.py --api-url http://localhost:8081/v1 --model Qwen/Qwen2.5-7B-Instruct --env BabyAI-GoToLocalS8N3-v0 --render
```

Optional arguments:

-   `--render`: Visualize the environment
-   `--verbose`: Print detailed information
-   `--save`: Save trajectory data and images

## Evaluation

Each environment will be evaluated with the following command:

```bash
python main.py --verbose --max-steps 100 --api-url http://localhost:8081/v1 --save --episodes 4 --timeout 360 --run-id <team_id> --model <model_name> --env <target_env>
```

For the submission, provide name of the model you are using.

## Development Tips

1. Use `play.py` to manually test and understand the environments
2. You can test your agent/prompting with OpenAI models first, but remember that local models won't be as capable
3. Consider these improvements for your agent:
    - Implement long-term memory of events and objects (can be explicit with code, or created by the model itself)
    - Add chain of thought reasoning, store previous thoughts in the context
    - Add more information about the environment rules in the prompt
    - Improve 'perception' - the way the information about objects and walls (!) is presented
    - Let the model output a high level plan every N steps
    - Parse multiple actions from model's output - this will be useful when using CoT approach

## Recommended Models

-   Qwen/Qwen2.5-7B-Instruct
    -   There are variants of different sizes
-   microsoft/phi-4
-   mistralai/Mistral-Nemo-Instruct-2407
    -   To run this model, add "--max_model_len 8000" to vllm command

## Final Notes

-   The challenge aims to create a general solution - don't overfit to the provided environments
-   For available environments, see: https://minigrid.farama.org/environments/babyai/
-   The hidden evaluation environments will test your agent's generalization capabilities
