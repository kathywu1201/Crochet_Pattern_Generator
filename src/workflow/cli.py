"""
CLI for orchestrating pipelines for image-to-vector, LLM-RAG, and LLM Chat.

Typical usage example from command line:
    python cli.py --pipeline
"""

import os
import argparse
import random
import string
from kfp import dsl
from kfp import compiler
import google.cloud.aiplatform as aip

# Environment Variables
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]
PIPELINE_ROOT = f"gs://{GCS_BUCKET_NAME}/pipeline_root"
GCS_SERVICE_ACCOUNT = os.environ["GCS_SERVICE_ACCOUNT"]
GCP_REGION = os.environ["GCP_REGION"]

# Component Images
IMAGE_2_VECTOR_IMAGE = "your-repo/image_2_vector"
LLM_RAG_IMAGE = "your-repo/llm-rag"
LLM_CHAT_IMAGE = "your-repo/llm-chat"

def generate_uuid(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def image_2_vector():
    print("image_2_vector()")

    @dsl.container_component
    def image_2_vector_component():
        return dsl.ContainerSpec(
            image=IMAGE_2_VECTOR_IMAGE,
            command=["python", "cli.py", "--process", "--upload"],
            args=[f"--bucket {GCS_BUCKET_NAME}"],
        )

    @dsl.pipeline
    def image_2_vector_pipeline():
        image_2_vector_component()

    compiler.Compiler().compile(
        image_2_vector_pipeline, package_path="image_2_vector.yaml"
    )

    aip.init(project=GCP_PROJECT, staging_bucket=f"gs://{GCS_BUCKET_NAME}")

    job_id = generate_uuid()
    job = aip.PipelineJob(
        display_name=f"image-2-vector-{job_id}",
        template_path="image_2_vector.yaml",
        pipeline_root=PIPELINE_ROOT,
    )
    job.run(service_account=GCS_SERVICE_ACCOUNT)


def llm_rag():
    print("llm_rag()")

    @dsl.container_component
    def llm_rag_component():
        return dsl.ContainerSpec(
            image=LLM_RAG_IMAGE,
            command=["python", "cli.py", "--chunk", "--embed", "--load"],
            args=[f"--bucket {GCS_BUCKET_NAME}"],
        )

    @dsl.pipeline
    def llm_rag_pipeline():
        llm_rag_component()

    compiler.Compiler().compile(llm_rag_pipeline, package_path="llm_rag.yaml")

    aip.init(project=GCP_PROJECT, staging_bucket=f"gs://{GCS_BUCKET_NAME}")

    job_id = generate_uuid()
    job = aip.PipelineJob(
        display_name=f"llm-rag-{job_id}",
        template_path="llm_rag.yaml",
        pipeline_root=PIPELINE_ROOT,
    )
    job.run(service_account=GCS_SERVICE_ACCOUNT)


def llm_chat():
    print("llm_chat()")

    @dsl.container_component
    def llm_chat_component():
        return dsl.ContainerSpec(
            image=LLM_CHAT_IMAGE,
            command=["python", "cli.py", "--chat"],
            args=[f"--bucket {GCS_BUCKET_NAME}"],
        )

    @dsl.pipeline
    def llm_chat_pipeline():
        llm_chat_component()

    compiler.Compiler().compile(llm_chat_pipeline, package_path="llm_chat.yaml")

    aip.init(project=GCP_PROJECT, staging_bucket=f"gs://{GCS_BUCKET_NAME}")

    job_id = generate_uuid()
    job = aip.PipelineJob(
        display_name=f"llm-chat-{job_id}",
        template_path="llm_chat.yaml",
        pipeline_root=PIPELINE_ROOT,
    )
    job.run(service_account=GCS_SERVICE_ACCOUNT)


def full_pipeline():
    print("full_pipeline()")

    @dsl.pipeline
    def ml_pipeline():
        # Image to Vector
        image_2_vector_task = (
            dsl.ContainerSpec(
                image=IMAGE_2_VECTOR_IMAGE,
                command=["python", "cli.py", "--process", "--upload"],
                args=[f"--bucket {GCS_BUCKET_NAME}"],
            )
            .set_display_name("Image to Vector")
        )

        # LLM RAG
        llm_rag_task = (
            dsl.ContainerSpec(
                image=LLM_RAG_IMAGE,
                command=["python", "cli.py", "--chunk", "--embed", "--load"],
                args=[f"--bucket {GCS_BUCKET_NAME}"],
            )
            .set_display_name("LLM RAG")
            .after(image_2_vector_task)
        )

        # LLM Chat
        llm_chat_task = (
            dsl.ContainerSpec(
                image=LLM_CHAT_IMAGE,
                command=["python", "cli.py", "--chat"],
                args=[f"--bucket {GCS_BUCKET_NAME}"],
            )
            .set_display_name("LLM Chat")
            .after(llm_rag_task)
        )

    compiler.Compiler().compile(ml_pipeline, package_path="pipeline.yaml")

    aip.init(project=GCP_PROJECT, staging_bucket=f"gs://{GCS_BUCKET_NAME}")

    job_id = generate_uuid()
    job = aip.PipelineJob(
        display_name=f"ml-pipeline-{job_id}",
        template_path="pipeline.yaml",
        pipeline_root=PIPELINE_ROOT,
    )
    job.run(service_account=GCS_SERVICE_ACCOUNT)


def main(args=None):
    if args.image_2_vector:
        image_2_vector()
    elif args.llm_rag:
        llm_rag()
    elif args.llm_chat:
        llm_chat()
    elif args.pipeline:
        full_pipeline()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Workflow CLI for ML Pipeline")

    parser.add_argument(
        "--image_2_vector", action="store_true", help="Run the Image to Vector step"
    )
    parser.add_argument("--llm_rag", action="store_true", help="Run the LLM RAG step")
    parser.add_argument("--llm_chat", action="store_true", help="Run the LLM Chat step")
    parser.add_argument("--pipeline", action="store_true", help="Run the full pipeline")

    args = parser.parse_args()
    main(args)
