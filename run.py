# Copyright Sierra

import os
import argparse
from dotenv import load_dotenv
from tau_bench.types import RunConfig
from tau_bench.run import run
from litellm import provider_list
from tau_bench.envs.user import UserStrategy

# 加载 .env 文件中的环境变量
load_dotenv()


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument(
        "--env", type=str, choices=["retail", "airline"], default="retail"
    )
    parser.add_argument(
        "--model",
        default='qwen3-14b-official-lightllm',
        type=str,
        help="The model to use for the agent",
    )
    parser.add_argument(
        "--model-provider",
        default='lightllm',
        type=str,
        choices=provider_list,
        help="The model provider for the agent",
    )
    parser.add_argument(
        "--user-model",
        type=str,
        default="gpt-4o-mini-2024-07-18",
        help="The model to use for the user simulator",
    )
    parser.add_argument(
        "--user-model-provider",
        default='azure',
        type=str,
        choices=provider_list,
        help="The model provider for the user simulator",
    )
    parser.add_argument(
        "--agent-strategy",
        type=str,
        default="tool-calling",
        choices=["tool-calling", "act", "react", "few-shot"],
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="The sampling temperature for the action model",
    )
    parser.add_argument(
        "--task-split",
        type=str,
        default="test",
        choices=["train", "test", "dev"],
        help="The split of tasks to run (only applies to the retail domain for now",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1, help="Run all tasks if -1")
    parser.add_argument("--task-ids", type=int, nargs="+", help="(Optional) run only the tasks with the given IDs")
    parser.add_argument("--log-dir", type=str, default="results")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Number of tasks to run in parallel",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--user-strategy", type=str, default="llm", choices=[item.value for item in UserStrategy])
    parser.add_argument("--few-shot-displays-path", type=str, help="Path to a jsonlines file containing few shot displays")
    # API相关参数
    parser.add_argument("--api_base", default="http://10.119.17.112:60012/generate", type=str, help="API base URL for the model provider")
    
    # 新增lightllm的参数
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter for LightLLM")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling parameter for LightLLM")
    parser.add_argument("--repetition_penalty", type=float, default=1.05, help="Repetition penalty parameter for LightLLM")
    parser.add_argument("--max_new_tokens", type=int, default=8192, help="Maximum new tokens for LightLLM")
    parser.add_argument("--do_sample", type=bool, default=True, help="Whether to use sampling for LightLLM")
    parser.add_argument("--skip_special_tokens", type=bool, default=False, help="Whether to skip special tokens for LightLLM")
    parser.add_argument("--add_special_tokens", type=bool, default=False, help="Whether to add special tokens for LightLLM")
    parser.add_argument("--stop_sequences", type=str, nargs="+", default=["<|im_end|>"], help="Stop sequences for LightLLM")
    parser.add_argument("--enable_thinking", type=bool, default=True, help="Whether to use thinking for LightLLM")
    args = parser.parse_args()
    print(args)
    return RunConfig(
        model_provider=args.model_provider,
        user_model_provider=args.user_model_provider,
        model=args.model,
        user_model=args.user_model,
        num_trials=args.num_trials,
        env=args.env,
        agent_strategy=args.agent_strategy,
        temperature=args.temperature,
        task_split=args.task_split,
        start_index=args.start_index,
        end_index=args.end_index,
        task_ids=args.task_ids,
        log_dir=args.log_dir,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        shuffle=args.shuffle,
        user_strategy=args.user_strategy,
        few_shot_displays_path=args.few_shot_displays_path,
        # API相关参数
        api_base=args.api_base,
        # LightLLM参数
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        skip_special_tokens=args.skip_special_tokens,
        add_special_tokens=args.add_special_tokens,
        stop_sequences=args.stop_sequences,
        enable_thinking=args.enable_thinking
    )


def main():
    config = parse_args()
    # print(f"run_config:{config}")
    run(config)


if __name__ == "__main__":
    main()
