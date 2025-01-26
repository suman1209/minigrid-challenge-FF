# MiniGrid Agent Challenge

Your task is to create an agent that can solve different grid-world environments using language models. The agent will be evaluated on 10 different environments - 5 provided for development and 5 hidden for final evaluation - all 10 will used for the final score.

### Evaluation Environments (Provided)

-   BabyAI-KeyInBox-v0
-   BabyAI-GoToLocalS8N3-v0
-   BabyAI-KeyCorridorS6R3-v0
-   BabyAI-PutNextS5N2-v0
-   BabyAI-SynthS5R2-v0

### Rules

1. You can only modify the `agent.py` file
2. Do not change the signature of the `Agent` constructor or `handle_state` function
3. Any local model can be used, as long as it runs on an A40 GPU
4. If the model runs out of memory during evaluation, the episode is lost - make sure you handle context window properly.
5. Each environment will be evaluated with 10 episodes
6. Maximum 100 steps per episode
7. 1-minute timeout per episode
8. Use of LLMs for development is allowed - try Claude with Sonnet 3.5, or Cursor IDE with different models

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
python main.py --verbose --max-steps 100 --api-url http://localhost:8081/v1 --model Qwen/Qwen2.5-7B-Instruct --env BabyAI-GoToLocalS8N3-v0 --render
```

Optional arguments:

-   `--render`: Visualize the environment
-   `--verbose`: Print detailed information
-   `--save`: Save trajectory data and images

## Evaluation

Each environment will be evaluated with the following command:

```bash
python main.py --verbose --max-steps 100 --api-url http://localhost:8081/v1 --save --episodes 10 --model model_name --env target_env
```

For the submission, provide name of the model you are using.

## Development Tips

1. Use `play.py` to manually test and understand the environments
2. You can test your agent/prompting with OpenAI models first, but remember that local models won't be as capable
3. Consider these improvements for your agent:
    - Add message history
    - Implement memory of events and objects
    - Add chain of thought reasoning
    - Adjust sampling parameters

## Recommended Models

-   Qwen/Qwen2.5-7B-Instruct
-   DeepSeek R1 distills (e.g., deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
    -   Note: This model outputs Chain of Thought by default
    -   You may need to adjust `max_tokens` when generating responses, and implement parsing function

## Final Notes

-   The challenge aims to create a general solution - don't overfit to the provided environments
-   For available environments, see: https://minigrid.farama.org/environments/babyai/
-   The hidden evaluation environments will test your agent's generalization capabilities
